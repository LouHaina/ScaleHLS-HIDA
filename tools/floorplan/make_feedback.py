#!/usr/bin/env python3
"""Turn one Vivado run into the feedback file of scalehls-physical-floorplan.

Inputs (all produced by the flow in tools/floorplan):
  --hier-util  report_utilization -hierarchical -hierarchical_depth 1 of the
               synthesized design (measured size per top-level cell)
  --nets       extract_nets.tcl output (top-level connectivity in bits)
  --rtl        HLS Verilog dir (maps nested instance names to their cell)
  --run        run_pnr.tcl output dir (summary.txt, paths_<stage>.csv)
  --prev       the pass's own output JSON of the iteration that was run
  --applied    comma list of RTL passes the run's netlist was built with

Output JSON:
  cells        {cell: {lut, ff, dsp, bram, uram}}
  edges        [{a, b, bits}]                     cell names
  critical     [{a, b, weight}]                   worst-path cell pairs, 1..2
  path_classes {class: count over the worst paths}
  previous     the previous assignment (node key -> slot)
  slot_mode, applied_rtl, wns, tns
"""
import argparse, csv, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_floorplan as gf  # noqa: E402

CLASSES = ('ctrl_broadcast', 'fsm_axi_gate', 'unit_internal', 'placement_xslr', 'placement_local')


def cell_of(pin, cells):
    head = pin.split('/')[0]
    return head if head in cells else 'top'


def classify(rows, cells):
    """Path classes (see tools/floorplan/iterate.py for the rationale)."""
    c = {k: 0 for k in CLASSES}
    for r in rows:
        s, e = cell_of(r['start'], cells), cell_of(r['end'], cells)
        logic, route = float(r['logic']), float(r['route'])
        if re.search(r'ap_CS_fsm', r['start']) and re.match(r'axi_\d+_m_axi_U', e):
            c['fsm_axi_gate'] += 1
        elif s != e and s.startswith('grp_') and e.startswith('grp_') and \
                re.search(r'loop_dataflow_output_count|ap_sync_reg|ap_done_reg', r['end']):
            c['ctrl_broadcast'] += 1
        elif s == e and logic >= route:
            c['unit_internal'] += 1
        else:
            c['placement_xslr' if r['inter_slr'] == '1' else 'placement_local'] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hier-util', required=True)
    ap.add_argument('--nets', required=True)
    ap.add_argument('--rtl')
    ap.add_argument('--run', required=True)
    ap.add_argument('--stage', default='place')
    ap.add_argument('--prev', required=True)
    ap.add_argument('--applied', default='')
    ap.add_argument('--top-paths', type=int, default=100)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    units = gf.read_units(a.hier_util)
    units.pop('forward', None)
    owner = gf.rtl_owner_map(a.rtl, units) if a.rtl else None
    edges = gf.read_edges(a.nets, units, owner)

    rows = list(csv.DictReader(open(os.path.join(a.run, f'paths_{a.stage}.csv'))))[:a.top_paths]
    crit = {}
    if rows:
        worst = min(float(r['slack']) for r in rows)
        best = max(float(r['slack']) for r in rows)
        for r in rows:
            s, e = cell_of(r['start'], units), cell_of(r['end'], units)
            if s == e:
                continue
            k = tuple(sorted((s, e)))
            crit[k] = crit.get(k, 0) + 1.0 + (best - float(r['slack'])) / max(best - worst, 1e-3)
    kv = dict(re.findall(r'(\w+)=(\S+)', open(os.path.join(a.run, 'summary.txt')).read()))
    prev = json.load(open(a.prev))
    fb = dict(
        cells={n: v for n, v in units.items()},
        edges=[dict(a=x, b=y, bits=w) for (x, y), w in sorted(edges.items())],
        critical=[dict(a=x, b=y, weight=round(w, 3)) for (x, y), w in sorted(crit.items())],
        path_classes=classify(rows, units),
        previous=prev['assignment'],
        slot_mode=prev.get('slot_mode', 'slr'),
        applied_rtl=[p for p in a.applied.split(',') if p],
        wns=float(kv.get(f'{a.stage}_wns', 'nan')),
        tns=float(kv.get(f'{a.stage}_tns', 'nan')),
    )
    json.dump(fb, open(a.out, 'w'), indent=1)
    print(f"feedback: {len(fb['cells'])} cells, {len(fb['edges'])} edges, {len(fb['critical'])} critical pairs, "
          f"classes {fb['path_classes']}, wns {fb['wns']}")


if __name__ == '__main__':
    main()
