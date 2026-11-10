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

def assignment_record(g, dev, K, units, edges, areas, assign, slack):
    """The assignment.json content shared by `solve` and `budget`."""
    full = dict(assign)
    for n in g['nodes']:
        if not n.get('is_unit') and n.get('parent') in assign: full[n['name']] = assign[n['parent']]
    channels = {}
    cut = []
    for e in edges:
        dd = abs(assign[e['src']] - assign[e['dst']])
        if dd == 0 or e['src'].startswith('__') or e['dst'].startswith('__'): continue
        cut.append(dict(e, distance=dd))
        if e['kind'] in ('token', 'stream') and e.get('channel'):
            channels[e['channel']] = {'depth': max(int(e.get('depth', 1)), 2) + 2 * dd + slack}
    real_units = [u for u in units if not u.startswith('__')]
    full = {n: k for n, k in full.items() if not n.startswith('__')}
    return dict(device=dev['name'], slrs=K, assignment=full, units=real_units, channels=channels,
                cut_edges=cut, score=ilp.score(assign, edges, areas, K), areas=areas)

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
    if axi_slr is not None and '__mem__' in units:
        anchors['__mem__'] = axi_slr   # memory subsystem pseudo-node; AXI edges pull units toward it
    groups = [(e['src'], e['dst']) for e in edges if e['kind'] == 'buffer' and not a.allow_buffer_cut]
    cycles = find_cycles(units, edges)   # before feedback adds control edges
    penalties, locks, fb_groups, ctrl_added = {}, {}, [], []
    if a.feedback:
        prev = json.load(open(a.prev)) if a.prev else {}
        prev_assign = prev.get('assignment', {})
        # Accumulate over every iteration so far: a pair that failed once keeps
        # its penalty, otherwise the solver can move it straight back.
        pen = defaultdict(float); times = defaultdict(int)
        for path in a.feedback:
            for pair, info in json.load(open(path)).get('failing_node_pairs', {}).items():
                s, d = pair.split('->')
                if s not in areas or d not in areas or s == d: continue
                key = tuple(sorted((s, d)))
                pen[key] += a.lam * abs(info['worst_slack']) * 1000; times[key] += 1
        implicated = set()
        for (s, d), p in pen.items():
            implicated.update((s, d))
            ids = [i for i, e in enumerate(edges) if {e['src'], e['dst']} == {s, d}]
            if not ids:
                # The failing path is a control handshake (ap_done/ap_sync through
                # the top FSM) that the dataflow graph does not contain. Without
                # an edge the penalty would be dropped and nothing would change.
                edges.append(dict(src=s, dst=d, width=1, kind='ctrl', channel='', depth=1))
                ids = [len(edges) - 1]; ctrl_added.append(f'{s}<->{d}')
            penalties[ids[0]] = penalties.get(ids[0], 0) + p
            if a.group_after and times[(s, d)] >= a.group_after: fb_groups.append((s, d))
        for u in units:
            if u in prev_assign and u not in implicated and a.lock: locks[u] = prev_assign[u]
        print(f'feedback: {len(pen)} failing unit pairs, {len(ctrl_added)} control edges added, '
              f'{len(fb_groups)} pairs grouped, {len(locks)} units locked')
    # Relax in order: locks keep incremental P&R reuse high, but they must not
    # stop a failing pair from being brought together. Take the first attempt
    # that leaves no penalized edge cut, else the one that cuts the fewest.
    attempts = [('as requested', locks, groups + fb_groups)]
    if locks: attempts.append(('without locks', {}, groups + fb_groups))
    if fb_groups: attempts.append(('without locks and feedback groups', {}, groups))
    assign, info, best_cut = None, None, None
    for label, lk, gr in attempts:
        asg, inf = ilp.solve(units, edges, areas, K, cap, dev['sll_per_boundary'] * a.sll_ratio,
                             alpha=1.0, beta=a.beta, penalties=penalties, anchors=anchors,
                             groups=gr, locks=lk, cycles=cycles, time_limit=a.time_limit)
        if asg is None:
            if info is None: info = inf
            continue
        still_cut = sum(1 for i in penalties if asg[edges[i]['src']] != asg[edges[i]['dst']])
        if best_cut is None or still_cut < best_cut:
            assign, info, best_cut = asg, inf, still_cut
            if label != 'as requested': print(f'note: solved {label}')
        if still_cut == 0: break
    if best_cut: print(f'warning: {best_cut} failing unit pairs are still split across SLRs (capacity or anchors prevent co-location)')
    if assign is None:
        print('ILP infeasible:', info, file=sys.stderr); sys.exit(2)
    out = assignment_record(g, dev, K, units, edges, areas, assign, a.slack)
    out['solver'] = dict(status=info['status'], seconds=round(info['seconds'], 2))
    json.dump(out, open(a.out, 'w'), indent=1)
    sc = out['score']
    print(f"solved {info['status']} in {info['seconds']:.1f}s: cut={sc['n_cut']} {sc['cut_edges']} bits/boundary={sc['sll_bits_per_boundary']}")
    for k, per in enumerate(sc['per_slr']): print(f"  SLR{k}: {per}")

# ----------------------------------------------------------------------------
# budget: per-unit unroll factors under a per-SLR resource budget
# ----------------------------------------------------------------------------
def slr_caps(dev, K, rho, rho_slr):
    rhos = [float(v) for v in rho_slr.split(',')] if rho_slr else [rho] * K
    if len(rhos) != K: raise SystemExit(f'--rho-slr needs {K} values')
    return [{r: dev['capacity_per_slr'].get(r, 0) * rhos[k] for r in RES} for k in range(K)], rhos

def ordered_units(g):
    """Unit names in compiler order (the order of floorplan-unit-factors)."""
    us = [n for n in g['nodes'] if n.get('is_unit') and not n['name'].startswith('__')]
    if any(n.get('unit_index', -1) < 0 for n in us):
        raise SystemExit('graph has no unit_index: export it with a compiler that supports unit factors')
    return [n['name'] for n in sorted(us, key=lambda n: n['unit_index'])]

def graph_config(g, calib=None):
    """One compiled design: unit -> (factor, areas, latency), in unit order."""
    _, _, areas = unit_graph(g, calib)
    out = {}
    for u in ordered_units(g):
        n = g['node_by_name'][u]
        if n.get('unroll', -1) <= 0:
            raise SystemExit(f'unit {u} has no recorded unroll factor (pass floorplan-graph-output to the pipeline so factors are recorded)')
        out[u] = (n['unroll'], areas[u], n.get('latency', 0))
    return out

def cmd_sweep(a):
    """Compile the design once per factor (all units at that factor) and once
    per value of the plain HIDA knob, exporting a graph each time."""
    os.makedirs(a.out_dir, exist_ok=True)
    def compile_to(name, extra):
        graph = os.path.join(a.out_dir, name)
        run(a.compile.format(graph=graph, extra=extra))
        return graph
    knob = [int(v) for v in a.knob.split(',') if v]
    factors = [int(v) for v in a.factors.split(',') if v]
    knob_graphs = [compile_to(f'knob_{F}.json', f'loop-unroll-factor={F}') for F in knob]
    n = len(ordered_units(load_graph(knob_graphs[0])))
    unit_graphs = []
    for f in factors:
        uf = os.path.join(a.out_dir, f'all_{f}.factors.json')
        json.dump({'unit_factors': [f] * n}, open(uf, 'w'))
        unit_graphs.append(compile_to(f'all_{f}.json', f'loop-unroll-factor={knob[0]} floorplan-unit-factors={uf}'))
    print(f'{n} units; {len(unit_graphs)} factor graphs and {len(knob_graphs)} knob graphs in {a.out_dir}')
    print('next: floorplan.py budget ' + ' '.join(unit_graphs) + ' --baseline ' + ' '.join(knob_graphs) + ' --device <device.json>')

def cmd_budget(a):
    import budget as bg
    dev = json.load(open(a.device)); K = a.slrs or dev['slrs']
    caps, rhos = slr_caps(dev, K, a.rho, a.rho_slr)
    graphs = [load_graph(p) for p in a.graphs]
    baselines = [(p, load_graph(p)) for p in (a.baseline or [])]
    order = ordered_units(graphs[0])
    samples = defaultdict(list)
    for g in graphs + [g for _, g in baselines]:
        cfg = graph_config(g)
        if list(cfg) != order:
            raise SystemExit('the graphs do not have the same units in the same order; they must come from the same model and options except the unroll factors')
        for u, v in cfg.items(): samples[u].append(v)
    tables = bg.build_tables(samples)
    g = graphs[0]
    units, edges, areas0 = unit_graph(g)
    fixed = {u: areas0[u] for u in units if u not in tables}
    anchors = {}
    axi_slr = dev.get('memory_anchors', {}).get('axi')
    if axi_slr is not None and '__mem__' in units: anchors['__mem__'] = axi_slr
    groups = [(e['src'], e['dst']) for e in edges if e['kind'] == 'buffer']
    prev = json.load(open(a.assignment))['assignment'] if a.assignment else None
    kw = dict(fixed=fixed, objective=a.objective, anchors=anchors, groups=groups, assignment=prev, time_limit=a.time_limit)
    # Solve a little below the budget: units that share an external buffer
    # influence each other and HIDA's output varies slightly between runs, so the
    # recompiled design is not exactly the sum of the measured units.
    # With --compile the chosen design is compiled and measured; if an SLR is
    # over its budget, the budget of that resource is tightened by the
    # overshoot and the choice is repeated.
    tighten = {r: 1.0 for r in RES}; final_graph = None; rounds = 0
    while True:
        solve_caps = [{r: v * (1 - a.margin) * tighten[r] for r, v in c.items()} for c in caps]
        factors, assign, info = bg.solve_budget(units, tables, edges, K, solve_caps, **kw)
        if factors is None:
            print('no combination of the compiled factors fits the budget:', info, file=sys.stderr); sys.exit(2)
        if not a.compile: break
        uf = a.out_factors + '.try.json'; final_graph = a.out_factors + '.final.json'
        json.dump({'unit_factors': [factors[u] for u in order]}, open(uf, 'w'))
        run(a.compile.format(graph=final_graph, extra=f'loop-unroll-factor=2 floorplan-unit-factors={uf}'))
        _, _, measured = unit_graph(load_graph(final_graph))
        worst = {}
        for k in range(K):
            for r in RES:
                used = sum(measured[u].get(r, 0) for u in units if assign[u] == k and u in measured)
                if caps[k].get(r, 0) > 0 and used > caps[k][r]: worst[r] = max(worst.get(r, 0), used / caps[k][r])
        if not worst: break
        rounds += 1
        if rounds > a.max_rounds:
            print(f'still over budget after {a.max_rounds} rounds: {worst}', file=sys.stderr); sys.exit(3)
        for r, over in worst.items(): tighten[r] *= 0.98 / over
        print(f'  round {rounds}: recompiled design is over budget ({", ".join(f"{r} by {100 * (v - 1):.1f}%" for r, v in worst.items())}); solving again')
    lsum, lmax = bg.latency(tables, factors)
    use = bg.usage(tables, factors, assign, K, fixed)
    print(f"budget solved {info['status']} in {info['seconds']}s; share of each SLR {rhos}; objective {a.objective}; "
          f"candidates per unit {sorted(tables[order[0]])}")
    for k in range(K):
        pct = ', '.join(f"{r} {use[k][r]} ({100 * use[k][r] / caps[k][r] * rhos[k]:.0f}%)" for r in RES if caps[k].get(r, 0) > 0 and use[k][r])
        print(f"  SLR{k}: {sum(1 for u in factors if assign[u] == k)} units; {pct}")
    print('  unit factors: ' + ' '.join(str(factors[u]) for u in order))
    print(f'  latency by the HIDA estimate: sum {lsum:.4g}, largest unit {lmax:.4g}')
    # Two references under the same budget: the plain HIDA knob (--baseline),
    # and the same factor for every unit (the input graphs themselves).
    key = 'latency_max' if a.objective == 'max' else 'latency_sum'
    mine = lmax if a.objective == 'max' else lsum
    def evaluate(label, pairs):
        rows = []
        for path, bgraph in pairs:
            cfg = graph_config(bgraph)
            ok = bg.fits(units, {u: (v[1], v[2]) for u, v in cfg.items()}, edges, K, caps, **kw) is not None
            bs, bm = sum(v[2] for v in cfg.values()), max(v[2] for v in cfg.values())
            tot = {r: int(sum(v[1].get(r, 0) for v in cfg.values())) for r in ('dsp', 'bram')}
            rows.append(dict(graph=os.path.basename(path), fits=ok, latency_sum=bs, latency_max=bm, **tot))
            print(f"  {label} {os.path.basename(path)}: {'fits' if ok else 'does not fit'}; dsp {tot['dsp']}, bram {tot['bram']}; latency sum {bs:.4g}, largest unit {bm:.4g}")
        ok_rows = [r for r in rows if r['fits']]
        if ok_rows:
            best = min(ok_rows, key=lambda r: r[key])
            print(f"  best {label} that fits: {best['graph']}; per-unit factors are {best[key] / mine:.2f}x faster by the estimate")
        return rows
    rows = evaluate('knob', baselines)
    uniform = evaluate('same factor', list(zip(a.graphs, graphs)))
    json.dump(dict(unit_factors=[factors[u] for u in order], units=order, rho=rhos, objective=a.objective,
                   predicted_latency_sum=lsum, predicted_latency_max=lmax, predicted_usage=use, baselines=rows, uniform=uniform, rounds=rounds),
              open(a.out_factors, 'w'), indent=1)
    areas = {u: {r: int(round(v)) for r, v in bg.area_at(tables, u, factors[u]).items()} for u in factors}
    areas.update(fixed)
    out = assignment_record(g, dev, K, units, edges, areas, assign, a.slack)
    out['solver'] = info; out['unit_factors'] = {u: factors[u] for u in order}
    out['predicted_latency'] = dict(sum=lsum, max=lmax); out['rounds'] = rounds
    if final_graph: print(f'  recompiled and within budget after {rounds} extra rounds: {final_graph}')
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'wrote {a.out_factors} (for floorplan-unit-factors=) and {a.out}')

def cmd_budget_check(a):
    """Compare a graph compiled with the chosen factors against the budget."""
    g = load_graph(a.graph); dev = json.load(open(a.device))
    asg = json.load(open(a.assignment)); K = asg['slrs']
    caps, rhos = slr_caps(dev, K, a.rho, a.rho_slr)
    units, _, areas = unit_graph(g, json.load(open(a.calibration)) if a.calibration else None)
    assign = asg['assignment']; over = False
    use = [{r: 0 for r in RES} for _ in range(K)]
    for u in units:
        if u in assign:
            for r in RES: use[assign[u]][r] += areas[u].get(r, 0)
    for k in range(K):
        parts = []
        for r in RES:
            if caps[k].get(r, 0) <= 0 or not use[k][r]: continue
            bad = use[k][r] > caps[k][r]; over |= bad
            parts.append(f"{r} {use[k][r]} of {int(caps[k][r])}{' OVER' if bad else ''}")
        print(f"  SLR{k}: {', '.join(parts)}")
    want = asg.get('unit_factors', {}); pred = asg.get('areas', {})
    wrong, off = [], []
    lats = []
    for u in units:
        n = g['node_by_name'][u]
        if u.startswith('__'): continue
        lats.append(n.get('latency', 0))
        if u in want and n.get('unroll', -1) != want[u]: wrong.append(f"{u}: compiled with factor {n.get('unroll')} instead of {want[u]}")
        p, m = pred.get(u, {}).get(a.resource, 0), areas[u].get(a.resource, 0)
        if p and abs(m - p) > a.tolerance * p: off.append(f"{u}: {a.resource} {m}, expected {p}")
    for r in wrong + off: print('  ' + r)
    pl = asg.get('predicted_latency')
    if lats and pl:
        print(f"  latency by the HIDA estimate: sum {sum(lats):.4g} (expected {pl['sum']:.4g}), largest unit {max(lats):.4g} (expected {pl['max']:.4g})")
    if wrong: print('the graph was not compiled with the chosen factors')
    print('budget exceeded: lower --rho (or --rho-slr for that SLR) and solve again' if over
          else f'within budget; {len(off)} units differ from the expected {a.resource} by more than {a.tolerance:.0%}')
    sys.exit(1 if over or wrong else 0)

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
    best = None; prev_assign = None; feedback = None; calib = None; feedbacks = []
    for it in range(cfg.get('iters', 4)):
        d = os.path.join(wd, f'iter_{it}'); os.makedirs(d, exist_ok=True)
        graph = os.path.join(d, 'graph.json'); asg = os.path.join(d, 'assignment.json'); cpp = os.path.join(d, 'design.cpp')
        # 1. export graph (first compile, no assignment)
        run(cfg['compile'].format(graph=graph, assignment='', cpp=os.path.join(d, 'design_nofp.cpp')))
        # 2. solve
        extra = f' --feedback {" ".join(feedbacks)} --prev {prev_assign}' if feedbacks and prev_assign else ''
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
        feedbacks.append(feedback)
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
    s.add_argument('--sll-ratio', type=float, default=0.5); s.add_argument('--beta', type=float, default=0.0, help='cost per percent of imbalance; 0 = cut only'); s.add_argument('--slack', type=int, default=2)
    s.add_argument('--allow-buffer-cut', action='store_true'); s.add_argument('--calibration'); s.add_argument('--feedback', nargs='+', help='feedback.json of every earlier iteration'); s.add_argument('--prev')
    s.add_argument('--group-after', type=int, default=2, help='force a unit pair into one SLR after it failed this many iterations (0 = never)')
    s.add_argument('--lam', type=float, default=1.0); s.add_argument('--lock', action='store_true', default=True); s.add_argument('--time-limit', type=float, default=60)
    s.set_defaults(fn=cmd_solve)
    e = sub.add_parser('emit'); e.add_argument('assignment'); e.add_argument('--device', required=True); e.add_argument('--out-tcl', default='floorplan.tcl')
    e.add_argument('--out-manifest', default='manifest.json'); e.add_argument('--soft', action='store_true'); e.set_defaults(fn=cmd_emit)
    c = sub.add_parser('calibrate'); c.add_argument('graph'); c.add_argument('--hls-dir', default='.'); c.add_argument('--report'); c.add_argument('--out', default='calibration.json'); c.set_defaults(fn=cmd_calibrate)
    l = sub.add_parser('loop'); l.add_argument('config'); l.set_defaults(fn=cmd_loop)
    w = sub.add_parser('sweep', help='compile the design at several unroll factors and export a graph each time')
    w.add_argument('--compile', required=True, help='compile command with {extra} inside the pipeline options and {graph} as floorplan-graph-output')
    w.add_argument('--factors', default='1,2,4,8,16,32', help='factors given to all units at once')
    w.add_argument('--knob', default='2,8,32,64', help='values of the plain loop-unroll-factor, for comparison')
    w.add_argument('--out-dir', default='sweep'); w.set_defaults(fn=cmd_sweep)
    b = sub.add_parser('budget', help='per-unit unroll factors under a per-SLR resource budget')
    b.add_argument('graphs', nargs='+', help='graphs of the same design compiled at different unit factors (from sweep)')
    b.add_argument('--baseline', nargs='+', help='graphs compiled with the plain loop-unroll-factor knob, for comparison')
    b.add_argument('--device', required=True); b.add_argument('--slrs', type=int, default=0)
    b.add_argument('--rho', type=float, default=0.6, help='share of each SLR that may be used')
    b.add_argument('--rho-slr', help='one share per SLR, e.g. 0.6,0.5,0.6 (lower the SLR that was congested)')
    b.add_argument('--margin', type=float, default=0.05, help='part of the budget kept free for differences between the measured units and the recompiled design')
    b.add_argument('--compile', help='compile command as for sweep; the chosen design is compiled and the choice repeated until it is within budget')
    b.add_argument('--max-rounds', type=int, default=5)
    b.add_argument('--objective', choices=['sum', 'max'], default='sum', help='sum of the unit latencies, or the largest one')
    b.add_argument('--assignment', help='keep the SLR assignment of this file and only choose factors')
    b.add_argument('--slack', type=int, default=2); b.add_argument('--time-limit', type=float, default=60)
    b.add_argument('--out', default='assignment.json'); b.add_argument('--out-factors', default='unit_factors.json')
    b.set_defaults(fn=cmd_budget)
    k = sub.add_parser('budget-check', help='check a recompiled graph against the budget')
    k.add_argument('graph'); k.add_argument('assignment'); k.add_argument('--device', required=True)
    k.add_argument('--rho', type=float, default=0.6); k.add_argument('--rho-slr'); k.add_argument('--calibration')
    k.add_argument('--resource', default='dsp'); k.add_argument('--tolerance', type=float, default=0.2)
    k.set_defaults(fn=cmd_budget_check)
    a = ap.parse_args(); a.fn(a)

if __name__ == '__main__':
    main()
