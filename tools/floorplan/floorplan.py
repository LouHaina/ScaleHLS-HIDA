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

def unit_samples(graphs, calibrations):
    """unit -> [(factor, areas, latency)] over all graphs; also the unit order."""
    samples = defaultdict(list); order = None
    for gi, g in enumerate(graphs):
        calib = calibrations[gi] if gi < len(calibrations) else None
        units, _, areas = unit_graph(g, calib)
        idx = {u: g['node_by_name'][u].get('unit_index', -1) for u in units if not u.startswith('__')}
        if any(v < 0 for v in idx.values()):
            raise SystemExit('graph has no unit_index: export it with a compiler that has the unit factor support')
        this = [u for u, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        if order is None: order = this
        elif this != order:
            raise SystemExit('the graphs do not have the same units in the same order; they must come from the same model and options except the unroll factors')
        for u in this:
            n = g['node_by_name'][u]
            if n.get('unroll', -1) <= 0:
                raise SystemExit(f'unit {u} has no recorded unroll factor (set floorplan-graph-output on the pipeline so factors are recorded)')
            samples[u].append((n['unroll'], areas[u], n.get('latency', 0)))
    return samples, order

def cmd_budget(a):
    import budget as bg
    graphs = [load_graph(p) for p in a.graphs]
    calibs = [json.load(open(p)) for p in (a.calibration or [])]
    dev = json.load(open(a.device)); K = a.slrs or dev['slrs']
    caps, rhos = slr_caps(dev, K, a.rho, a.rho_slr)
    samples, order = unit_samples(graphs, calibs)
    models, no_latency = bg.fit_models(samples)
    g = graphs[0]
    units, edges, areas0 = unit_graph(g, calibs[0] if calibs else None)
    fixed = {u: areas0[u] for u in units if u not in models}
    base = {u: samples[u][0][0] for u in order}
    cands = {}
    for u in order:
        hi = max(max(models[u]['points']) * a.max_scale, 1)
        cands[u] = sorted(set(bg.candidate_factors(1, int(hi))) | set(models[u]['points']))
    anchors = {}
    axi_slr = dev.get('memory_anchors', {}).get('axi')
    if axi_slr is not None and '__mem__' in units: anchors['__mem__'] = axi_slr
    groups = [(e['src'], e['dst']) for e in edges if e['kind'] == 'buffer']
    prev = json.load(open(a.assignment))['assignment'] if a.assignment else None
    kw = dict(fixed=fixed, objective=a.objective, anchors=anchors, groups=groups, assignment=prev, time_limit=a.time_limit)
    factors, assign, info = bg.solve_budget(units, models, edges, K, caps, candidates=cands, **kw)
    if factors is None:
        print('no factors fit the budget even at the smallest candidates:', info, file=sys.stderr); sys.exit(2)
    # What the single HIDA knob reaches under the same budget, for comparison.
    scales = sorted({c / 8 for c in bg.candidate_factors(1, int(8 * a.max_scale))})
    us, uf, uasg = bg.best_uniform(units, models, edges, K, caps, base, scales, **kw)
    lat = bg.total_latency(models, factors, a.objective)
    print(f"budget solved {info['status']} in {info['seconds']}s, rho per SLR {rhos}, objective {a.objective}")
    if no_latency:
        print(f'note: {no_latency} units have no latency estimate; their share of the work is taken from the base factors')
    if len(graphs) == 1:
        print('note: one graph only, so area is assumed proportional to the factor (constant for BRAM/URAM); give graphs of two factors for a fitted model')
    use = bg.usage(models, factors, assign, K, fixed)
    for k in range(K):
        pct = ', '.join(f"{r} {use[k][r]} ({100 * use[k][r] / caps[k][r] * rhos[k]:.0f}%)" for r in RES if caps[k].get(r, 0) > 0 and use[k][r])
        print(f"  SLR{k}: {sum(1 for u in factors if assign[u] == k)} units; {pct}")
    print('  unit factors (base -> chosen): ' + ', '.join(f"{u.split('_')[-1]} {base[u]}->{factors[u]}" for u in order))
    cmp = None
    if uf is not None:
        ulat = bg.total_latency(models, uf, a.objective)
        cmp = dict(scale=us, latency=ulat, factors=[uf[u] for u in order])
        print(f"  predicted latency: {lat:.4g} with per-unit factors, {ulat:.4g} with one common scale (x{us:g} of the base); ratio {ulat / lat:.2f}x")
    else:
        print('  one common scale does not fit the budget at any tried size; only per-unit factors do')
    json.dump(dict(unit_factors=[factors[u] for u in order], units=order, base_factors=[base[u] for u in order],
                   rho=rhos, objective=a.objective, predicted_latency=lat, uniform=cmp,
                   predicted_usage=use), open(a.out_factors, 'w'), indent=1)
    areas = {u: {r: int(round(v)) for r, v in bg.area_at(models[u], factors[u]).items()} for u in factors}
    areas.update(fixed)
    out = assignment_record(g, dev, K, units, edges, areas, assign, a.slack)
    out['solver'] = info; out['unit_factors'] = {u: factors[u] for u in order}
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
            parts.append(f"{r} {use[k][r]} / {int(caps[k][r])}{' OVER' if bad else ''}")
        print(f"  SLR{k}: {', '.join(parts)}")
    want = asg.get('unit_factors', {}); pred = asg.get('areas', {})
    rows = []
    for u in units:
        n = g['node_by_name'][u]
        if u in want and n.get('unroll', -1) != want[u]: rows.append(f"{u}: factor {n.get('unroll')} instead of {want[u]}")
        p, m = pred.get(u, {}).get(a.resource, 0), areas[u].get(a.resource, 0)
        if p and abs(m - p) > a.tolerance * p: rows.append(f"{u}: {a.resource} {m} vs predicted {p}")
    for r in rows: print('  ' + r)
    print('budget exceeded: pass this graph to `budget` as another point and solve again' if over
          else f'within budget; {len(rows)} units differ from the prediction by more than {a.tolerance:.0%}')
    sys.exit(1 if over else 0)

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
    b = sub.add_parser('budget', help='per-unit unroll factors under a per-SLR resource budget')
    b.add_argument('graphs', nargs='+', help='graph.json of the same design at one or more unroll factors; the first is the base')
    b.add_argument('--device', required=True); b.add_argument('--slrs', type=int, default=0)
    b.add_argument('--rho', type=float, default=0.6, help='share of each SLR that may be used')
    b.add_argument('--rho-slr', help='one share per SLR, e.g. 0.6,0.5,0.6 (lower the SLR that was congested)')
    b.add_argument('--objective', choices=['sum', 'max'], default='sum', help='sum: latency of one frame; max: slowest unit')
    b.add_argument('--max-scale', type=float, default=4.0, help='largest factor tried, relative to the largest compiled one')
    b.add_argument('--calibration', nargs='+', help='calibration.json per graph, in the same order')
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
