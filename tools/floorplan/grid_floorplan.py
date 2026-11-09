#!/usr/bin/env python3
"""Grid-level ILP floorplanner (iteration on PR #4's proto_floorplan.py).

Differences from proto_floorplan.py, each motivated by the Vivado evaluation of
PR #4 on ResNet18 (WNS: no floorplan +1.9 ns, PR #4 ILP -3.1 ns):

  * Graph from the synthesized netlist (extract_graph.tcl), not from the emitted
    C++.  In HIDA designs the top-level nodes share no C++ arguments; they talk
    only through the top-level dataflow controller (ap_start/ap_done/continue),
    so the C++ graph has no edges and the ILP degenerates to area balancing.
  * Measured per-unit area (report_utilization -hierarchical), not proxies.
  * Slots are clock regions (8 x 12 on xcvu37p), not SLRs.  The objective is
    Manhattan distance (in clock regions) of the connections plus a large
    penalty per SLR boundary crossed, so a design that fits one SLR is never
    split, and connected units are packed around their common controller.
  * Feedback (PR #4 doc, Sec. 8): the start/end units of the worst paths of the
    previous routed run (analyze_routed.tcl -> paths.csv) become extra edges
    weighted by criticality, so the next iteration pulls those pairs together.

Output: a Vivado Tcl with one pblock per used clock region and the unit cells
assigned to it, plus assignment.json.
"""
import argparse, csv, json, math, re, time
from collections import defaultdict


def read_units(hier_rpt):
    """Units and their measured area from report_utilization -hierarchical."""
    units = {}
    hdr = None
    for line in open(hier_rpt, errors='replace'):
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells[0] == 'Instance':
            hdr = cells
            continue
        if hdr is None or not line.startswith('|   ') or line.startswith('|    '):
            continue  # depth-1 rows only
        row = dict(zip(hdr, cells))
        name = row['Instance']
        if name == '(forward)':
            name = 'top'
        units[name] = dict(lut=int(row['Total LUTs']), ff=int(row['FFs']),
                           dsp=int(row['DSP Blocks']),
                           bram=int(row['RAMB36']) + (int(row['RAMB18']) + 1) // 2,
                           uram=int(row['URAM']))
    return units


def read_edges(nets_csv, units):
    """Top-level connectivity -> weighted unit pairs.

    Accepts nets.csv from extract_graph.tcl (one row per net, ';'-separated
    sinks, weight 1 per net) or from extract_nets.tcl (one row per unit pair
    with a 'bits' column).  Names that are not depth-1 cells (flattened
    top-level glue) are folded into 'top'."""
    w = defaultdict(float)
    canon = lambda n: n if n in units else 'top'
    for r in csv.DictReader(open(nets_csv)):
        d = canon(r['driver'])
        bits = float(r.get('bits') or 1)
        for s in r['sinks'].split(';'):
            s = canon(s)
            if s != d:
                w[tuple(sorted((d, s)))] += bits
    return w


def read_feedback(paths_csv, units, top_n):
    """Worst-path start/end units -> criticality-weighted extra edges."""
    rows = list(csv.DictReader(open(paths_csv)))[:top_n]
    if not rows:
        return {}
    worst = min(float(r['slack']) for r in rows)
    best = max(float(r['slack']) for r in rows)
    fb = defaultdict(float)

    def unit(pin):
        head = pin.split('/')[0]
        return head if head in units else 'top'
    for r in rows:
        a, b = unit(r['start']), unit(r['end'])
        if a == b:
            continue
        crit = 1.0 + (best - float(r['slack'])) / max(best - worst, 1e-3)  # 1..2
        fb[tuple(sorted((a, b)))] += crit
    return fb


def read_crs(cr_csv):
    crs = {}
    for r in csv.DictReader(open(cr_csv)):
        crs[r['cr']] = dict(x=int(r['x']), y=int(r['y']), lut=8 * int(r['slice']),
                            ff=16 * int(r['slice']), dsp=int(r['dsp']),
                            bram=int(r['ramb36']), uram=int(r['uram']))
    if all(v['bram'] == 0 for v in crs.values()):
        # extract_graph.tcl matched no RAMB36 site type on this device; use the
        # device average (xcvu37p: 2016 RAMB36 tiles over 96 clock regions).
        for v in crs.values():
            v['bram'] = 21
    return crs


def slr_slots(crs, rows_per_slr):
    """Aggregate clock regions into one slot per SLR (x=0, y=first row)."""
    out = {}
    for c, v in crs.items():
        k = v['y'] // rows_per_slr
        s = out.setdefault(f'SLR{k}', dict(x=0, y=k * rows_per_slr, lut=0, ff=0, dsp=0, bram=0, uram=0))
        for r in ('lut', 'ff', 'dsp', 'bram', 'uram'):
            s[r] += v[r]
    return out


def solve(units, edges, crs, rows_per_slr, rho, slr_pen, region, time_limit, free,
          balance=0.0, pin=None, prev=None, stay=0.0):
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    nodes = [u for u in units if u not in free]
    slots = [c for c in crs if region is None or crs[c]['y'] in region]
    y = {(n, s): m.NewBoolVar(f'y_{n}_{s}') for n in nodes for s in slots}
    X, Y, S = {}, {}, {}
    ymax = max(crs[s]['y'] for s in slots)
    for n in nodes:
        m.AddExactlyOne(y[n, s] for s in slots)
        X[n] = m.NewIntVar(0, 7, f'x_{n}')
        Y[n] = m.NewIntVar(0, ymax, f'yy_{n}')
        S[n] = m.NewIntVar(0, ymax // rows_per_slr, f's_{n}')
        m.Add(X[n] == sum(crs[s]['x'] * y[n, s] for s in slots))
        m.Add(Y[n] == sum(crs[s]['y'] * y[n, s] for s in slots))
        m.Add(S[n] == sum((crs[s]['y'] // rows_per_slr) * y[n, s] for s in slots))
    for s in slots:
        for r in ('lut', 'ff', 'dsp', 'bram', 'uram'):
            load = [units[n][r] * y[n, s] for n in nodes if units[n][r] > 0]
            if load:
                m.Add(sum(load) <= int(rho * crs[s][r]))
    for n, s in (pin or {}).items():
        if n in nodes:
            m.Add(y[n, s] == 1)
    cost = []
    if balance > 0:
        # minimise the largest per-slot utilisation (max over LUT and DSP), in 1/1000
        mx = m.NewIntVar(0, 1000, 'maxutil')
        for s in slots:
            for r in ('lut', 'dsp'):
                if crs[s][r] > 0:
                    m.Add(sum(int(1000 * units[n][r]) * y[n, s] for n in nodes if units[n][r] > 0)
                          <= mx * crs[s][r])
        cost.append(int(balance) * mx)
    if prev and stay > 0:
        # stability (iterative loop): moving a node away from its slot of the
        # previous iteration costs stay * its LUT share, so ties and mirror
        # images do not reshuffle the design between iterations
        tot = sum(units[n]['lut'] for n in nodes) or 1
        for n in nodes:
            if prev.get(n) in slots:
                c = int(round(stay * 1000 * max(units[n]['lut'], 1) / tot)) or 1
                cost.append(c * (1 - y[n, prev[n]]))
    for (a, b), w in edges.items():
        if a not in X or b not in X:
            continue
        dx = m.NewIntVar(0, 7, ''); dy = m.NewIntVar(0, ymax, ''); ds = m.NewIntVar(0, 2, '')
        m.AddAbsEquality(dx, X[a] - X[b]); m.AddAbsEquality(dy, Y[a] - Y[b])
        m.AddAbsEquality(ds, S[a] - S[b])
        wi = int(round(10 * w))
        cost += [wi * dx, wi * dy, wi * slr_pen * ds]
    m.Minimize(sum(cost))
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = time_limit
    sol.parameters.num_workers = 8
    t0 = time.time(); st = sol.Solve(m)
    status = sol.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise SystemExit(f'no solution: {status}')
    asg = {n: next(s for s in slots if sol.Value(y[n, s])) for n in nodes}
    return asg, status, sol.ObjectiveValue(), time.time() - t0


def band_slots(crs, rows):
    """Aggregate clock regions into horizontal bands of <rows> rows (B<first row>)."""
    out = {}
    for c, v in crs.items():
        y0 = v['y'] // rows * rows
        s = out.setdefault(f'B{y0}', dict(x=0, y=y0, rows=rows, lut=0, ff=0, dsp=0, bram=0, uram=0))
        for r in ('lut', 'ff', 'dsp', 'bram', 'uram'):
            s[r] += v[r]
    return out


def pblock_range(s, crs, rows_per_slr, band_rows=2):
    if s.startswith('B'):
        lo = int(s[1:]); hi = lo + band_rows - 1
        xmin = min(v['x'] for v in crs.values()); xmax = max(v['x'] for v in crs.values())
        return f'CLOCKREGION_X{xmin}Y{lo}:CLOCKREGION_X{xmax}Y{hi}'
    if s.startswith('SLR'):
        k = int(s[3:]); lo, hi = k * rows_per_slr, k * rows_per_slr + rows_per_slr - 1
        xmin = min(v['x'] for v in crs.values()); xmax = max(v['x'] for v in crs.values())
        return f'CLOCKREGION_X{xmin}Y{lo}:CLOCKREGION_X{xmax}Y{hi}'
    return f'CLOCKREGION_{s}:CLOCKREGION_{s}'


def emit_tcl(asg, path, contain_routing, crs_all=None, band_rows=2):
    by = defaultdict(list)
    for n, s in asg.items():
        by[s].append(n)
    L = ['# Grid ILP floorplan (tools/floorplan/grid_floorplan.py)',
         'catch {delete_pblocks [get_pblocks]}']
    for s, ns in sorted(by.items()):
        pb = f'pb_{s}'
        L += [f'create_pblock {pb}', f'resize_pblock {pb} -add {{{pblock_range(s, crs_all, 4, band_rows)}}}',
              f'set_property CONTAIN_ROUTING {str(contain_routing).lower()} [get_pblocks {pb}]',
              f'set_property SNAPPING_MODE ON [get_pblocks {pb}]', 'set cells [list]']
        for n in sorted(ns):
            if n == 'top':
                L.append('lappend cells {*}[get_cells -filter {IS_PRIMITIVE==1}]')
                continue
            L += [f'set c [get_cells -quiet {{{n}}}]',
                  f'if {{[llength $c] != 1}} {{ error "floorplan: {n} matched [llength $c] cells" }}',
                  'lappend cells {*}$c']
        L += [f'add_cells_to_pblock {pb} $cells']
    open(path, 'w').write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hier-util', required=True)
    ap.add_argument('--nets', required=True)
    ap.add_argument('--crs', required=True)
    ap.add_argument('--feedback', help='paths.csv of a previous routed run')
    ap.add_argument('--feedback-top', type=int, default=100)
    ap.add_argument('--feedback-weight', type=float, default=1.0)
    ap.add_argument('--rho', type=float, default=0.6)
    ap.add_argument('--slr-pen', type=int, default=20)
    ap.add_argument('--rows', help='allowed clock-region rows, e.g. 4-7')
    ap.add_argument('--free', default='', help='comma list of units left unconstrained')
    ap.add_argument('--contain-routing', action='store_true')
    ap.add_argument('--slot-mode', choices=['cr', 'slr', 'band'], default='cr')
    ap.add_argument('--band-rows', type=int, default=2)
    ap.add_argument('--balance', type=float, default=0.0, help='weight of max per-slot utilisation')
    ap.add_argument('--pin', default='', help='unit=slot,... fixed assignments (e.g. top=SLR1)')
    ap.add_argument('--strip', help='clock-region columns x0-x1 the design is confined to')
    ap.add_argument('--prev', help='assignment json of the previous iteration (stability term)')
    ap.add_argument('--stay', type=float, default=1000.0, help='weight of the stability term')
    ap.add_argument('--time', type=float, default=120)
    ap.add_argument('--out-tcl', required=True)
    ap.add_argument('--out-json', required=True)
    a = ap.parse_args()
    units = read_units(a.hier_util)
    units.pop('forward', None)
    edges = read_edges(a.nets, units)
    if a.feedback:
        for k, v in read_feedback(a.feedback, units, a.feedback_top).items():
            edges[k] += a.feedback_weight * v
    crs_all = read_crs(a.crs)
    if a.strip:
        x0, x1 = map(int, a.strip.split('-'))
        crs_all = {c: v for c, v in crs_all.items() if x0 <= v['x'] <= x1}
    crs = {'slr': lambda: slr_slots(crs_all, 4), 'band': lambda: band_slots(crs_all, a.band_rows),
           'cr': lambda: crs_all}[a.slot_mode]()
    pin = dict(kv.split('=') for kv in a.pin.split(',') if kv)
    region = None
    if a.rows:
        lo, hi = map(int, a.rows.split('-')); region = set(range(lo, hi + 1))
    free = set(f for f in a.free.split(',') if f)
    asg, st, obj, dt = solve(units, edges, crs, 4, a.rho, a.slr_pen, region, a.time, free,
                             a.balance, pin,
                             json.load(open(a.prev))['assignment'] if a.prev else None, a.stay)
    emit_tcl(asg, a.out_tcl, a.contain_routing, crs_all, a.band_rows)
    json.dump(dict(assignment=asg, edges={f'{k[0]}|{k[1]}': v for k, v in edges.items()},
                   status=st, objective=obj), open(a.out_json, 'w'), indent=1)
    used = sorted(set(asg.values()), key=lambda s: (crs[s]['y'], crs[s]['x']))
    print(f'status={st} obj={obj:.0f} solve={dt:.1f}s units={len(asg)} edges={len(edges)} slots_used={len(used)}: {" ".join(used)}')


if __name__ == '__main__':
    main()
