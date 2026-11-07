#!/usr/bin/env python3
"""Floorplan driver for the floorplan-aware HIDA flow.

Subcommands:
  solve      graph.json -> assignment.json         (ILP over units)
  emit       assignment.json + graph.json -> floorplan.tcl + manifest.json
  calibrate  Vitis HLS csynth report -> calibration.json (measured areas)
  loop       run the iterative compile / HLS / Vivado loop from a config

graph.json comes from `scalehls-opt -scalehls-export-dataflow-graph` (or from
proto_floorplan.py --export-graph for C++-only experiments).
"""
import argparse, json, os, re, subprocess, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ilp  # noqa: E402

RES = ilp.RES

# ----------------------------------------------------------------------------
# graph helpers
# ----------------------------------------------------------------------------
def load_graph(path):
    g = json.load(open(path))
    g['node_by_name'] = {n['name']: n for n in g['nodes']}
    return g

def unit_graph(g, calibration=None):
    """Collapse leaves into their units. Returns (units, edges, areas, groups)."""
    nb = g['node_by_name']
    units = [n['name'] for n in g['nodes'] if n.get('is_unit')]
    unit_of = {n['name']: (n['name'] if n.get('is_unit') else n.get('parent')) for n in g['nodes']}
    areas = {}
    for u in units:
        n = nb[u]
        a = {'lut': n.get('lut', 0), 'ff': n.get('ff', n.get('lut', 0)), 'dsp': n.get('dsp', 0),
             'bram': n.get('bram', 0), 'uram': n.get('uram', 0)}
        for m in g['nodes']:          # HIDA: dsp of a unit already includes callees; bram/lut do not
            if m.get('parent') == u:
                a['bram'] += m.get('bram', 0); a['lut'] += m.get('lut', 0); a['ff'] += m.get('ff', m.get('lut', 0))
        areas[u] = a
    if calibration:
        for u, meas in calibration.get('areas', {}).items():
            if u in areas:
                for r in RES:
                    if r in meas: areas[u][r] = meas[r]
    edges = []
    for e in g['edges']:
        a, b = unit_of.get(e['src']), unit_of.get(e['dst'])
        if not a or not b or a == b: continue
        edges.append(dict(src=a, dst=b, width=e['width'], kind=e['kind'], channel=e.get('channel', ''), depth=e.get('depth', 1)))
    return units, edges, areas

def find_cycles(units, edges):
    """SCCs of the directed unit graph; returns [(edge indices, tokens)]."""
    adj = defaultdict(list)
    for i, e in enumerate(edges): adj[e['src']].append((e['dst'], i))
    index = {}; low = {}; stack = []; on = set(); sccs = []; counter = [0]
    def strong(v):
        index[v] = low[v] = counter[0]; counter[0] += 1; stack.append(v); on.add(v)
        for w, _ in adj[v]:
            if w not in index: strong(w); low[v] = min(low[v], low[w])
            elif w in on: low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop(); on.discard(w); comp.add(w)
                if w == v: break
            if len(comp) > 1: sccs.append(comp)
    sys.setrecursionlimit(10000)
    for u in units:
        if u not in index: strong(u)
    out = []
    for comp in sccs:
        ids = [i for i, e in enumerate(edges) if e['src'] in comp and e['dst'] in comp]
        tokens = sum(max(int(edges[i].get('depth', 1)), 1) for i in ids)
        out.append((ids, tokens))
    return out

# ----------------------------------------------------------------------------
# solve
# ----------------------------------------------------------------------------
def cmd_solve(a):
    g = load_graph(a.graph)
    dev = json.load(open(a.device))
    K = a.slrs or dev['slrs']
    calib = json.load(open(a.calibration)) if a.calibration else None
    units, edges, areas = unit_graph(g, calib)
    cap = {r: dev['capacity_per_slr'].get(r, 0) * a.rho for r in RES}
    if a.balance is not None:   # tight balance mode for designs far below capacity
        for r in RES:
            tot = sum(areas[u].get(r, 0) for u in units)
            if tot: cap[r] = min(cap[r], tot * (1 + a.balance) / K)
    anchors = {}
    axi_slr = dev.get('memory_anchors', {}).get('axi')
    if axi_slr is not None:
        for e in edges:
            if e['kind'] == 'axi': anchors.setdefault(e['src'], axi_slr)
    groups = [(e['src'], e['dst']) for e in edges if e['kind'] == 'buffer' and not a.allow_buffer_cut]
    penalties, locks = {}, {}
    if a.feedback:
        fb = json.load(open(a.feedback))
        prev = json.load(open(a.prev)) if a.prev else {}
        prev_assign = prev.get('assignment', {})
        failing = fb.get('failing_node_pairs', {})
        implicated = set()
        for pair, info in failing.items():
            s, d = pair.split('->')
            implicated.update((s, d))
            for i, e in enumerate(edges):
                if {e['src'], e['dst']} == {s, d}:
                    penalties[i] = penalties.get(i, 0) + a.lam * abs(info['worst_slack']) * 1000
        for u in units:
            if u in prev_assign and u not in implicated and a.lock: locks[u] = prev_assign[u]
    assign, info = ilp.solve(units, edges, areas, K, cap, dev['sll_per_boundary'] * a.sll_ratio,
                             alpha=1.0, beta=a.beta, penalties=penalties, anchors=anchors,
                             groups=groups, locks=locks, cycles=find_cycles(units, edges),
                             time_limit=a.time_limit)
    if assign is None:
        print('ILP infeasible:', info, file=sys.stderr); sys.exit(2)
    full = dict(assign)
    for n in g['nodes']:
        if not n.get('is_unit') and n.get('parent') in assign: full[n['name']] = assign[n['parent']]
    channels = {}
    cut = []
    for i, e in enumerate(edges):
        dd = abs(assign[e['src']] - assign[e['dst']])
        if dd == 0: continue
        cut.append(dict(e, distance=dd))
        if e['kind'] in ('token', 'stream') and e.get('channel'):
            channels[e['channel']] = {'depth': max(int(e.get('depth', 1)), 2) + 2 * dd + a.slack}
    out = dict(device=dev['name'], slrs=K, assignment=full, units=units, channels=channels,
               cut_edges=cut, score=ilp.score(assign, edges, areas, K), solver=dict(status=info['status'], seconds=round(info['seconds'], 2)),
               areas=areas)
    json.dump(out, open(a.out, 'w'), indent=1)
    sc = out['score']
    print(f"solved {info['status']} in {info['seconds']:.1f}s: cut={sc['n_cut']} {sc['cut_edges']} bits/boundary={sc['sll_bits_per_boundary']}")
    for k, per in enumerate(sc['per_slr']): print(f"  SLR{k}: {per}")

# ----------------------------------------------------------------------------
# emit
# ----------------------------------------------------------------------------
def unit_regexes(unit):
    u = re.escape(unit)
    return [rf'(.*/)?grp_{u}(_\d+)?_fu_\d+', rf'(.*/)?{u}(_\d+)?_U0', rf'(.*/)?{u}(_\d+)?_s_U0']

def cmd_emit(a):
    asg = json.load(open(a.assignment)); dev = json.load(open(a.device))
    K = asg['slrs']; units = asg['units']; assign = asg['assignment']
    manifest = {u: dict(slr=assign[u], regex=unit_regexes(u), expected=asg.get('areas', {}).get(u, {})) for u in units}
    L = []
    L.append('# Floorplan constraints generated by tools/floorplan/floorplan.py')
    L.append(f"# device: {dev['name']}  slrs: {K}  mode: {'soft' if a.soft else 'pblock'}")
    L.append('proc fp_unit_cells {patterns} {')
    L.append('  set cells [list]')
    L.append('  foreach p $patterns { foreach c [get_cells -hierarchical -quiet -regexp -filter [format {NAME =~ "%s"} $p]] { lappend cells $c } }')
    L.append('  return [lsort -unique $cells]')
    L.append('}')
    L.append('set fp_missing [list]')
    if not a.soft:
        L.append('catch {delete_pblocks [get_pblocks pblock_slr*]}')
        for k in range(K):
            rng = dev['pblock_ranges'].get(str(k), f'SLR{k}')
            L.append(f'create_pblock pblock_slr{k}')
            L.append(f'resize_pblock pblock_slr{k} -add {{{rng}}}')
            L.append(f'set_property SNAPPING_MODE ON [get_pblocks pblock_slr{k}]')
    for u in units:
        pats = ' '.join('{' + p + '}' for p in manifest[u]['regex'])
        L.append(f'set c [fp_unit_cells [list {pats}]]')
        L.append(f'if {{[llength $c] == 0}} {{ lappend fp_missing {u} }} else {{')
        L.append(f'  puts "unit {u}: [llength $c] cells -> SLR{assign[u]}"')
        if a.soft:
            L.append(f'  set_property USER_SLR_ASSIGNMENT SLR{assign[u]} $c')
        else:
            L.append(f'  add_cells_to_pblock pblock_slr{assign[u]} $c')
            L.append(f'  set_property KEEP_HIERARCHY SOFT $c')
        L.append('}')
    # best-effort: keep the FIFO of every cut stream next to its consumer
    L.append('proc fp_pin_fifo {src_pats dst_pats slr} {')
    L.append('  set s [fp_unit_cells $src_pats]; set d [fp_unit_cells $dst_pats]')
    L.append('  if {[llength $s] == 0 || [llength $d] == 0} { return }')
    L.append('  set nets [get_nets -quiet -of_objects [get_pins -quiet -of_objects $s -filter {DIRECTION == OUT}]]')
    L.append('  foreach f [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects $nets] -filter {REF_NAME =~ *fifo* || NAME =~ *_U}] {')
    L.append('    if {[get_property PARENT $f] ne ""} { catch { add_cells_to_pblock pblock_slr$slr [get_cells [get_property PARENT $f]] } }')
    L.append('  }')
    L.append('  foreach n $nets { catch { set_property USER_CROSSING_SLR 1 $n } }')
    L.append('}')
    if not a.soft:
        for e in asg.get('cut_edges', []):
            if e['kind'] in ('token', 'stream'):
                sp = ' '.join('{' + p + '}' for p in unit_regexes(e['src'])); dp = ' '.join('{' + p + '}' for p in unit_regexes(e['dst']))
                L.append(f'fp_pin_fifo [list {sp}] [list {dp}] {assign[e["dst"]]}')
    L.append('if {[llength $fp_missing] > 0} { puts "WARNING: floorplan units with no matching cells: $fp_missing" }')
    L.append('puts "=== floorplan constraints applied ==="')
    open(a.out_tcl, 'w').write('\n'.join(L) + '\n')
    json.dump(dict(device=dev['name'], units=manifest, cut_edges=asg.get('cut_edges', [])), open(a.out_manifest, 'w'), indent=1)
    print(f'wrote {a.out_tcl} and {a.out_manifest} ({len(units)} units, {len(asg.get("cut_edges", []))} cut edges)')

# ----------------------------------------------------------------------------
# calibrate
# ----------------------------------------------------------------------------
def parse_csynth_instances(path):
    """Instance table of a Vitis HLS csynth report: module -> areas."""
    txt = open(path, errors='replace').read()
    sec = txt.split('* Instance:', 1)
    if len(sec) < 2: return {}
    rows = {}
    cols = None
    for line in sec[1].splitlines():
        if line.strip().startswith('+'): continue
        if not line.strip().startswith('|'): 
            if cols and rows: break
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells[0] == 'Instance': cols = cells; continue
        if not cols or len(cells) != len(cols) or cells[0] in ('Total',): continue
        rec = {}
        for c, v in zip(cols, cells):
            key = {'BRAM_18K': 'bram', 'DSP': 'dsp', 'DSP48E': 'dsp', 'FF': 'ff', 'LUT': 'lut', 'URAM': 'uram'}.get(c)
            if key:
                try: rec[key] = int(v)
                except ValueError: pass
        rows[cells[1]] = rec
    return rows

def cmd_calibrate(a):
    g = load_graph(a.graph)
    top = g['top']
    report = a.report or os.path.join(a.hls_dir, f'{top}_csynth.rpt')
    inst = parse_csynth_instances(report)
    areas = {}
    for n in g['nodes']:
        name = n['name']
        for mod, rec in inst.items():
            base = re.sub(r'(_s)?(_\d+)?$', '', mod)
            if base == name or base == f'{top}_{name}':
                cur = areas.setdefault(name, {})
                for r, v in rec.items(): cur[r] = cur.get(r, 0) + v
    scale = {}
    for n in g['nodes']:
        if n['name'] in areas and n.get('dsp'):
            scale[n['name']] = round(areas[n['name']].get('dsp', 0) / n['dsp'], 3)
    json.dump(dict(report=report, areas=areas, dsp_scale=scale), open(a.out, 'w'), indent=1)
    print(f'calibrated {len(areas)} of {len(g["nodes"])} nodes from {report}')

# ----------------------------------------------------------------------------
# loop
# ----------------------------------------------------------------------------
def run(cmd, cwd=None):
    print('+', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if r.returncode: raise SystemExit(f'command failed ({r.returncode}): {cmd}')

def cmd_loop(a):
    """Config keys: workdir, device, slrs, period, iters, compile (template with {graph} {assignment} {cpp}),
    hls (template with {cpp} {hls_dir}), vivado (template with {dcp} {tcl} {period} {out} {prev}), python."""
    cfg = json.load(open(a.config)); wd = cfg['workdir']; os.makedirs(wd, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__)); py = cfg.get('python', sys.executable)
    best = None; prev_assign = None; feedback = None; calib = None
    for it in range(cfg.get('iters', 4)):
        d = os.path.join(wd, f'iter_{it}'); os.makedirs(d, exist_ok=True)
        graph = os.path.join(d, 'graph.json'); asg = os.path.join(d, 'assignment.json'); cpp = os.path.join(d, 'design.cpp')
        # 1. export graph (first compile, no assignment)
        run(cfg['compile'].format(graph=graph, assignment='', cpp=os.path.join(d, 'design_nofp.cpp')))
        # 2. solve
        extra = f' --feedback {feedback} --prev {prev_assign}' if feedback and prev_assign else ''
        extra += f' --calibration {calib}' if calib else ''
        run(f"{py} {here}/floorplan.py solve {graph} --device {cfg['device']} --slrs {cfg.get('slrs', 0)} --out {asg}{extra}")
        # 3. compile again with the assignment, then emit constraints
        run(cfg['compile'].format(graph=graph, assignment=asg, cpp=cpp))
        run(f"{py} {here}/floorplan.py emit {asg} --device {cfg['device']} --out-tcl {d}/floorplan.tcl --out-manifest {d}/manifest.json")
        # 4. HLS (only when the C++ changed) and Vivado
        hls_dir = os.path.join(d, 'hls')
        if it == 0 or open(cpp).read() != open(os.path.join(wd, f'iter_{it-1}', 'design.cpp')).read():
            run(cfg['hls'].format(cpp=cpp, hls_dir=hls_dir))
            if calib is None:
                calib = os.path.join(d, 'calibration.json')
                run(f"{py} {here}/floorplan.py calibrate {graph} --hls-dir {hls_dir} --out {calib}")
                run(f"{py} {here}/floorplan.py solve {graph} --device {cfg['device']} --slrs {cfg.get('slrs', 0)} --out {asg} --calibration {calib}{extra}")
                run(cfg['compile'].format(graph=graph, assignment=asg, cpp=cpp))
                run(f"{py} {here}/floorplan.py emit {asg} --device {cfg['device']} --out-tcl {d}/floorplan.tcl --out-manifest {d}/manifest.json")
        else:
            hls_dir = os.path.join(wd, f'iter_{it-1}', 'hls')
        out = os.path.join(d, 'out'); prev_dcp = os.path.join(wd, f'iter_{it-1}', 'out', 'post_route.dcp') if it else ''
        run(cfg['vivado'].format(dcp=cfg['dcp'].format(hls_dir=hls_dir), tcl=f'{d}/floorplan.tcl', period=cfg['period'], out=out, prev=prev_dcp))
        # 5. feedback
        feedback = os.path.join(d, 'feedback.json')
        run(f"{py} {here}/parse_reports.py {out} --out {feedback}")
        fb = json.load(open(feedback)); wns = fb.get('timing', {}).get('wns')
        print(f'### iteration {it}: WNS={wns} failing pairs={list(fb.get("failing_node_pairs", {}))[:5]}')
        if best is None or (wns is not None and wns > best[1]): best = (it, wns)
        if wns is not None and wns >= 0: print('timing closed'); break
        prev_assign = asg
    print(f'best iteration: {best}')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('solve'); s.add_argument('graph'); s.add_argument('--device', required=True); s.add_argument('--slrs', type=int, default=0)
    s.add_argument('--out', default='assignment.json'); s.add_argument('--rho', type=float, default=0.7); s.add_argument('--balance', type=float, default=None)
    s.add_argument('--sll-ratio', type=float, default=0.5); s.add_argument('--beta', type=float, default=1.0); s.add_argument('--slack', type=int, default=2)
    s.add_argument('--allow-buffer-cut', action='store_true'); s.add_argument('--calibration'); s.add_argument('--feedback'); s.add_argument('--prev')
    s.add_argument('--lam', type=float, default=1.0); s.add_argument('--lock', action='store_true', default=True); s.add_argument('--time-limit', type=float, default=60)
    s.set_defaults(fn=cmd_solve)
    e = sub.add_parser('emit'); e.add_argument('assignment'); e.add_argument('--device', required=True); e.add_argument('--out-tcl', default='floorplan.tcl')
    e.add_argument('--out-manifest', default='manifest.json'); e.add_argument('--soft', action='store_true'); e.set_defaults(fn=cmd_emit)
    c = sub.add_parser('calibrate'); c.add_argument('graph'); c.add_argument('--hls-dir', default='.'); c.add_argument('--report'); c.add_argument('--out', default='calibration.json'); c.set_defaults(fn=cmd_calibrate)
    l = sub.add_parser('loop'); l.add_argument('config'); l.set_defaults(fn=cmd_loop)
    a = ap.parse_args(); a.fn(a)

if __name__ == '__main__':
    main()
