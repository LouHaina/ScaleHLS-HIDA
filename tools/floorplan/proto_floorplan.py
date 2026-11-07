#!/usr/bin/env python3
"""Prototype of the SLR floorplanner described in docs/FloorplanAwareHIDA-Design.md.

It works directly on HIDA-emitted HLS C++ so the algorithm can be exercised
without a compiler build or Vivado:

  1. parse the C++ into a two-level hierarchy (top -> units -> leaves),
  2. recover the dataflow edges from shared arguments between calls,
  3. derive proxy areas per function (DSP from unrolled multiplies, BRAM from
     local arrays and partition factors, LUT from statement count),
  4. solve the SLR assignment ILP with OR-Tools CP-SAT,
  5. score any assignment (ILP or an existing floorplan_constraints.tcl) with
     the same cost model.

Usage:
  proto_floorplan.py design.cpp --top forward --slrs 3 [--baseline-tcl f.tcl]
"""
import argparse, math, re, sys, time, json
from collections import defaultdict, OrderedDict

# ----------------------------------------------------------------------------
# 1. Parse the generated C++
# ----------------------------------------------------------------------------
FUNC_RE = re.compile(r'^void (\w+)\($', re.M)
CALL_RE = re.compile(r'^\s+(\w+)\((.*)\);', re.M)
ARG_RE = re.compile(r'^\s*(?:hls::stream<(\w+)>\s*&(\w+)|(ap_int<\d+>|ap_uint<\d+>|int|bool|float)\s+(\w+)((?:\[\d+\])*))\s*,?$')
LOCAL_RE = re.compile(r'^\s+(ap_int<(\d+)>|ap_uint<(\d+)>|int|bool|float)\s+(\w+)((?:\[\d+\])+);', re.M)
PART_RE = re.compile(r'#pragma HLS array_partition variable=(\w+)(?: cyclic| block)? factor=(\d+)')
DIM_RE = re.compile(r'\[(\d+)\]')

def elem_bits(t):
    m = re.search(r'<(\d+)>', t)
    if m: return int(m.group(1))
    return {'int': 32, 'bool': 1, 'float': 32}.get(t, 32)

def parse_cpp(path):
    src = open(path).read()
    funcs = OrderedDict()
    for m in FUNC_RE.finditer(src):
        name = m.group(1)
        start = m.end()
        hdr_end = src.index('\n) {', start)
        header = src[start:hdr_end]
        body_start = hdr_end + 4
        body_end = src.index('\n}\n', body_start)
        body = src[body_start:body_end]
        args = OrderedDict()
        for line in header.splitlines():
            a = ARG_RE.match(line)
            if not a: continue
            if a.group(2):
                args[a.group(2)] = {'kind': 'stream', 'bits': elem_bits(a.group(1)), 'dims': []}
            else:
                dims = [int(d) for d in DIM_RE.findall(a.group(5) or '')]
                args[a.group(4)] = {'kind': 'array' if dims else 'scalar',
                                    'bits': elem_bits(a.group(3)), 'dims': dims}
        locals_ = OrderedDict()
        for sm in re.finditer(r'^\s+hls::stream<(\w+)>\s+(\w+);', body, re.M):
            locals_[sm.group(2)] = {'kind': 'stream', 'bits': elem_bits(sm.group(1)), 'dims': []}
        for lm in LOCAL_RE.finditer(body):
            dims = [int(d) for d in DIM_RE.findall(lm.group(5))]
            locals_[lm.group(4)] = {'kind': 'array', 'bits': elem_bits(lm.group(1)), 'dims': dims}
        part = defaultdict(lambda: 1)
        for pm in PART_RE.finditer(body):
            part[pm.group(1)] *= int(pm.group(2))
        calls = [(c.group(1), [s.strip() for s in c.group(2).split(',') if s.strip()])
                 for c in CALL_RE.finditer(body) if c.group(1) in ('forward',) or c.group(1).startswith(('forward_node',))]
        stmts = [l.split('//')[0].rstrip() for l in body.splitlines()]
        stmts = [l for l in stmts if l.endswith(';') and '#pragma' not in l]
        muls = sum(l.count(' * ') for l in stmts if 'ap_int' in l or '=' in l)
        funcs[name] = dict(args=args, locals=locals_, part=part, calls=calls,
                           n_stmts=len(stmts), n_mul=muls,
                           inline='#pragma HLS inline\n' in body or '#pragma HLS inline' in body.split('\n', 3)[0:3].__str__(),
                           dataflow='#pragma HLS dataflow' in body)
    return funcs

# ----------------------------------------------------------------------------
# 2./3. Hierarchy, areas, edges
# ----------------------------------------------------------------------------
def own_area(f):
    bram = 0
    for n, a in f['locals'].items():
        bits = a['bits'] * math.prod(a['dims'])
        p = f['part'][n]
        bram += p * math.ceil(bits / p / 18000)
    return dict(lut=30 * f['n_stmts'], ff=30 * f['n_stmts'], dsp=f['n_mul'], bram=bram, uram=0)

def total_area(funcs, name, memo):
    if name in memo: return memo[name]
    a = own_area(funcs[name])
    for callee, _ in funcs[name]['calls']:
        if callee in funcs:
            for k, v in total_area(funcs, callee, memo).items(): a[k] += v
    memo[name] = a
    return a

def build_graph(funcs, top):
    """Return units (top-level callees), leaves per unit, and edges."""
    memo = {}
    units = [c for c, _ in funcs[top]['calls']]
    leaves = {u: [c for c, _ in funcs[u]['calls'] if c in funcs] or [u] for u in units}
    parent = {}
    for u in units:
        parent[u] = u
        for l in leaves[u]: parent[l] = u
    edges = []  # (a, b, width_bits, kind)
    def edges_in(fname, scope_kind):
        f = funcs[fname]
        users = defaultdict(list)
        for i, (callee, args) in enumerate(f['calls']):
            for a in args:
                users[a].append(callee)
        for a, us in users.items():
            if len(us) < 2: continue
            if a in f['locals'] and f['locals'][a]['kind'] == 'stream':
                info = f['locals'][a]; kind, width = 'token', info['bits'] + 2
            elif a in f['locals']:
                info = f['locals'][a]; kind = 'buffer'
                width = info['bits'] * f['part'][a] + 16 * f['part'][a]   # data + address per partition
            elif a in f['args']:
                info = f['args'][a]
                if info['kind'] == 'stream': kind, width = 'token', info['bits'] + 2
                elif info['kind'] == 'array': kind, width = scope_kind, info['bits'] + 20
                else: continue
            else:
                continue
            for i in range(len(us)):
                for j in range(i + 1, len(us)):
                    if us[i] != us[j]: edges.append((us[i], us[j], width, kind))
    edges_in(top, 'port')
    for u in units: edges_in(u, 'buffer')
    areas = {n: total_area(funcs, n, memo) if n in units else own_area(funcs[n]) for n in parent}
    return units, leaves, parent, edges, areas

# ----------------------------------------------------------------------------
# 4. ILP (CP-SAT)
# ----------------------------------------------------------------------------
KIND_W = {'stream': 1.0, 'buffer': 4.0, 'token': 0.1, 'port': 1.0}

def solve_ilp(nodes, edges, areas, K, cap, sll_cap, alpha=0.0, adjacency=True, time_limit=60, fixed=None):
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    idx = {n: i for i, n in enumerate(nodes)}
    y = {(n, k): m.NewBoolVar(f'y_{n}_{k}') for n in nodes for k in range(K)}
    p = {}
    for n in nodes:
        m.AddExactlyOne(y[n, k] for k in range(K))
        p[n] = m.NewIntVar(0, K - 1, f'p_{n}')
        m.Add(p[n] == sum(k * y[n, k] for k in range(K)))
    if fixed:
        for n, k in fixed.items(): m.Add(y[n, k] == 1)
    # capacity
    for k in range(K):
        for r in ('lut', 'ff', 'dsp', 'bram', 'uram'):
            if cap[r] > 0:
                m.Add(sum(int(areas[n][r]) * y[n, k] for n in nodes) <= int(cap[r]))
    # edges
    d, cost = {}, []
    boundary_load = defaultdict(list)
    for i, (a, b, w, kind) in enumerate(edges):
        if a not in idx or b not in idx: continue
        d[i] = m.NewIntVar(0, K - 1, f'd{i}')
        m.Add(d[i] >= p[a] - p[b]); m.Add(d[i] >= p[b] - p[a])
        if adjacency and kind in ('buffer', 'stream', 'token'): m.Add(d[i] <= 1)
        cost.append(int(round(w * KIND_W[kind] * 10)) * d[i])
        # crossing of boundary bb (between SLR bb and bb+1): lo <= bb < hi
        for bb in range(K - 1):
            a_low = m.NewBoolVar(f'al{i}_{bb}'); b_low = m.NewBoolVar(f'bl{i}_{bb}')
            m.Add(p[a] <= bb).OnlyEnforceIf(a_low); m.Add(p[a] > bb).OnlyEnforceIf(a_low.Not())
            m.Add(p[b] <= bb).OnlyEnforceIf(b_low); m.Add(p[b] > bb).OnlyEnforceIf(b_low.Not())
            c = m.NewBoolVar(f'c{i}_{bb}')
            m.AddBoolXOr([a_low, b_low, c.Not()])   # c = a_low xor b_low
            boundary_load[bb].append(int(w) * c)
    for bb, terms in boundary_load.items():
        m.Add(sum(terms) <= int(sll_cap))
    # imbalance (LUT proxy)
    obj = sum(cost)
    if alpha > 0:
        tot = sum(int(areas[n]['lut']) for n in nodes)
        mx = m.NewIntVar(0, tot, 'mx'); mn = m.NewIntVar(0, tot, 'mn')
        for k in range(K):
            s = sum(int(areas[n]['lut']) * y[n, k] for n in nodes)
            m.Add(mx >= s); m.Add(mn <= s)
        obj = obj + int(alpha) * (mx - mn)
    m.Minimize(obj)
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = time_limit; s.parameters.num_workers = 4
    t0 = time.time(); st = s.Solve(m); dt = time.time() - t0
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, dt, s.StatusName(st)
    return {n: s.Value(p[n]) for n in nodes}, dt, s.StatusName(st)

# ----------------------------------------------------------------------------
# 5. Scoring
# ----------------------------------------------------------------------------
def score(assign, edges, areas, K):
    cut_w = 0.0; cut_edges = defaultdict(int); bound = defaultdict(int); n_cut = 0
    for a, b, w, kind in edges:
        if a not in assign or b not in assign: continue
        dd = abs(assign[a] - assign[b])
        if dd:
            n_cut += 1; cut_edges[kind] += 1; cut_w += w * KIND_W[kind] * dd
            for bb in range(min(assign[a], assign[b]), max(assign[a], assign[b])): bound[bb] += w
    per = [dict(lut=0, dsp=0, bram=0) for _ in range(K)]
    for n, k in assign.items():
        for r in per[k]: per[k][r] += areas[n][r]
    return dict(weighted_cut=cut_w, cut_edges=dict(cut_edges), n_cut=n_cut,
                sll_bits_per_boundary=dict(bound), per_slr=per)

def parse_baseline_tcl(path):
    assign = {}; cur = None
    for line in open(path):
        m = re.search(r'set partition_(\d+)_cells', line)
        if m: cur = int(m.group(1))
        m = re.search(r'get_func_cells (\w+)', line)
        if m and cur is not None and line.startswith('set _cells'): assign[m.group(1)] = cur
        m = re.search(r'NAME =~ \*(\w+)\*', line)
        if m and cur is not None: assign[m.group(1)] = cur
    return assign

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cpp'); ap.add_argument('--top', default='forward'); ap.add_argument('--slrs', type=int, default=3)
    ap.add_argument('--baseline-tcl'); ap.add_argument('--balance', type=float, default=0.3,
                    help='each SLR may hold at most (1+balance)/K of every resource')
    ap.add_argument('--sll-cap', type=int, default=23040 // 2); ap.add_argument('--json')
    ap.add_argument('--export-graph', help='write the graph in the schema of scalehls-export-dataflow-graph')
    a = ap.parse_args()
    funcs = parse_cpp(a.cpp)
    units, leaves, parent, edges, areas = build_graph(funcs, a.top)
    all_leaves = [l for u in units for l in leaves[u]]
    print(f'functions={len(funcs)} units={len(units)} leaves={len(all_leaves)} edges={len(edges)} '
          f'(buffer={sum(e[3]=="buffer" for e in edges)}, port={sum(e[3]=="port" for e in edges)}, token={sum(e[3]=="token" for e in edges)})')
    tot = {r: sum(areas[u][r] for u in units) for r in ('lut', 'ff', 'dsp', 'bram', 'uram')}
    print('proxy totals:', tot)
    K = a.slrs
    cap = {r: math.ceil(tot[r] * (1 + a.balance) / K) for r in tot}
    results = {}
    # (a) unit level
    asg_u, dt, st = solve_ilp(units, [e for e in edges if e[0] in parent and e[1] in parent and parent[e[0]] != parent[e[1]] or e[3] != 'buffer'],
                              areas, K, cap, a.sll_cap, alpha=1)
    if asg_u:
        full = {l: asg_u[parent[l]] for l in all_leaves}; full.update(asg_u)
        results['ilp_unit_level'] = score(full, edges, areas, K); results['ilp_unit_level'].update(solve_s=round(dt, 2), status=st)
    # (b) leaf level, buffer edges may be cut (weight x4)
    leaf_areas = {l: areas[l] for l in all_leaves}
    unit_only = [u for u in units if leaves[u] == [u]]
    for u in unit_only: leaf_areas[u] = areas[u]
    leaf_nodes = list(dict.fromkeys(all_leaves + unit_only))
    # port/token edges are between units: map to the unit's first leaf as proxy anchor
    leaf_edges = []
    for x, y_, w, kind in edges:
        lx = x if x in leaf_areas else leaves[x][0]; ly = y_ if y_ in leaf_areas else leaves[y_][0]
        leaf_edges.append((lx, ly, w, kind))
    asg_l, dt, st = solve_ilp(leaf_nodes, leaf_edges, leaf_areas, K, cap, a.sll_cap, alpha=1)
    if asg_l:
        results['ilp_leaf_level'] = score(asg_l, leaf_edges, leaf_areas, K); results['ilp_leaf_level'].update(solve_s=round(dt, 2), status=st)
    # (c) baseline from an existing Tcl
    if a.baseline_tcl:
        base = parse_baseline_tcl(a.baseline_tcl)
        base = {n: k for n, k in base.items() if n in leaf_areas}
        missing = [n for n in leaf_nodes if n not in base]
        for n in missing:  # tiny leaves were skipped by the pass; put them with a sibling
            sib = [s for s in leaves[parent[n]] if s in base]
            base[n] = base[sib[0]] if sib else 0
        results['baseline_bfs_fm'] = score(base, leaf_edges, leaf_areas, K)
        results['baseline_bfs_fm']['unassigned_in_tcl'] = len(missing)
        split_units = sum(len({base[l] for l in leaves[u] if l in base}) > 1 for u in units)
        results['baseline_bfs_fm']['units_split_across_slrs'] = split_units
    for name, r in results.items():
        print(f'\n== {name}')
        for k, v in r.items(): print(f'  {k}: {v}')
    if a.json:
        json.dump(dict(units=units, leaves=leaves, edges=edges, areas=areas, results=results), open(a.json, 'w'), indent=1)
    if a.export_graph:
        nodes = []
        for u in units:
            nodes.append(dict(name=u, parent='', is_unit=True, inline=False, dataflow=True, latency=0, interval=0,
                              callees=leaves[u] if leaves[u] != [u] else [], **{r: areas[u][r] for r in ('lut', 'ff', 'dsp', 'bram', 'uram')}))
            for l in leaves[u]:
                if l == u: continue
                nodes.append(dict(name=l, parent=u, is_unit=False, inline=funcs[l]['inline'], dataflow=False, latency=0, interval=0,
                                  callees=[], lut=0, ff=0, dsp=0, bram=0, uram=0))  # unit areas already include leaves
        gedges = [dict(src=x, dst=y, kind=k, width=w, depth=1, scope=parent.get(x, a.top), channel=f'{parent.get(x, a.top)}:{i}' if k == 'token' else '')
                  for i, (x, y, w, k) in enumerate(edges)]
        json.dump(dict(top=a.top, resource_note='proxy areas from C++', nodes=nodes, edges=gedges), open(a.export_graph, 'w'), indent=1)

if __name__ == '__main__':
    main()
