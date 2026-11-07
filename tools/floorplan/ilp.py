"""SLR assignment ILP (CP-SAT) for the floorplan-aware HIDA flow.

Model (docs/FloorplanAwareHIDA-Design.md, section 6):
  y[v,k]   one-hot SLR of node v;  p[v] = sum k*y[v,k]
  d[e]     = |p[src] - p[dst]|  boundaries crossed by edge e
  min  sum_e w_e*d_e + alpha*sum_e pen_e*d_e + beta*imbalance
  s.t. capacity per SLR and resource, SLL budget per boundary, anchors,
       grouping, adjacency (d_e <= 1 for stream/token edges), cycle budget,
       locks from a previous iteration.
"""
import time
from collections import defaultdict

KIND_W = {'stream': 1.0, 'buffer': 4.0, 'token': 0.1, 'port': 1.0, 'dram': 0.5, 'axi': 0.0, 'other': 1.0}
RES = ('lut', 'ff', 'dsp', 'bram', 'uram')


def solve(nodes, edges, areas, K, cap, sll_cap, *, alpha=1.0, beta=1.0, penalties=None,
          anchors=None, groups=None, locks=None, cycles=None, adjacency=True,
          time_limit=60.0, workers=4):
    """nodes: list of names; edges: list of dict(src,dst,width,kind[,id]);
    areas: name -> {lut,ff,dsp,bram,uram}; cap: {res: per-SLR cap};
    penalties: edge index -> penalty weight; anchors: name -> slr;
    groups: list of (a, b) that must share an SLR; locks: name -> slr;
    cycles: list of (edge index list, tokens). Returns (assignment, info)."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    nodeset = set(nodes)
    y = {(n, k): m.NewBoolVar(f'y_{n}_{k}') for n in nodes for k in range(K)}
    p = {}
    for n in nodes:
        m.AddExactlyOne(y[n, k] for k in range(K))
        p[n] = m.NewIntVar(0, K - 1, f'p_{n}')
        m.Add(p[n] == sum(k * y[n, k] for k in range(K)))
    for n, k in (anchors or {}).items():
        if n in nodeset: m.Add(y[n, k] == 1)
    for n, k in (locks or {}).items():
        if n in nodeset: m.Add(y[n, k] == 1)
    for a, b in (groups or []):
        if a in nodeset and b in nodeset: m.Add(p[a] == p[b])
    for k in range(K):
        for r in RES:
            if cap.get(r, 0) > 0:
                m.Add(sum(int(areas[n].get(r, 0)) * y[n, k] for n in nodes) <= int(cap[r]))
    d, cost = {}, []
    boundary_load = defaultdict(list)
    penalties = penalties or {}
    for i, e in enumerate(edges):
        a, b = e['src'], e['dst']
        if a not in nodeset or b not in nodeset or a == b: continue
        d[i] = m.NewIntVar(0, K - 1, f'd{i}')
        m.Add(d[i] >= p[a] - p[b]); m.Add(d[i] >= p[b] - p[a])
        if adjacency and e['kind'] in ('buffer', 'stream', 'token'): m.Add(d[i] <= 1)
        w = e['width'] * KIND_W.get(e['kind'], 1.0) + alpha * penalties.get(i, 0.0)
        cost.append(int(round(w * 10)) * d[i])
        for bb in range(K - 1):
            a_low = m.NewBoolVar(f'al{i}_{bb}'); b_low = m.NewBoolVar(f'bl{i}_{bb}')
            m.Add(p[a] <= bb).OnlyEnforceIf(a_low); m.Add(p[a] > bb).OnlyEnforceIf(a_low.Not())
            m.Add(p[b] <= bb).OnlyEnforceIf(b_low); m.Add(p[b] > bb).OnlyEnforceIf(b_low.Not())
            c = m.NewBoolVar(f'c{i}_{bb}')
            m.AddBoolXOr([a_low, b_low, c.Not()])
            boundary_load[bb].append(int(e['width']) * c)
    for bb, terms in boundary_load.items():
        m.Add(sum(terms) <= int(sll_cap))
    for edge_ids, tokens in (cycles or []):
        ids = [i for i in edge_ids if i in d]
        if ids: m.Add(2 * sum(d[i] for i in ids) <= max(int(tokens) - 1, 0))
    obj = sum(cost)
    if beta > 0:
        bal_res = 'lut' if any(areas[n].get('lut', 0) for n in nodes) else 'dsp'
        tot = sum(int(areas[n].get(bal_res, 0)) for n in nodes)
        if tot > 0:
            mx = m.NewIntVar(0, tot, 'mx'); mn = m.NewIntVar(0, tot, 'mn')
            for k in range(K):
                s = sum(int(areas[n].get(bal_res, 0)) * y[n, k] for n in nodes)
                m.Add(mx >= s); m.Add(mn <= s)
            obj = obj + int(round(beta * 10)) * (mx - mn)
    m.Minimize(obj)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    t0 = time.time(); st = s.Solve(m); dt = time.time() - t0
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, dict(status=s.StatusName(st), seconds=dt)
    assign = {n: s.Value(p[n]) for n in nodes}
    return assign, dict(status=s.StatusName(st), seconds=dt, objective=s.ObjectiveValue(),
                        distances={i: s.Value(v) for i, v in d.items()})


def score(assign, edges, areas, K):
    cut_w = 0.0; cut_kinds = defaultdict(int); bound = defaultdict(int); n_cut = 0
    for e in edges:
        a, b = e['src'], e['dst']
        if a not in assign or b not in assign: continue
        dd = abs(assign[a] - assign[b])
        if dd:
            n_cut += 1; cut_kinds[e['kind']] += 1; cut_w += e['width'] * KIND_W.get(e['kind'], 1.0) * dd
            for bb in range(min(assign[a], assign[b]), max(assign[a], assign[b])): bound[bb] += e['width']
    per = [{r: 0 for r in RES} for _ in range(K)]
    for n, k in assign.items():
        for r in RES: per[k][r] += areas.get(n, {}).get(r, 0)
    return dict(weighted_cut=cut_w, cut_edges=dict(cut_kinds), n_cut=n_cut,
                sll_bits_per_boundary={str(k): v for k, v in bound.items()}, per_slr=per)
