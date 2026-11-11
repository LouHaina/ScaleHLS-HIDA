"""Per-SLR resource budget: choose the unroll factor of every unit.

Plain HIDA has one knob, loop-unroll-factor, and splits it over the units by
their share of the computation. Here every unit (top-level dataflow node) gets
its own factor, chosen together with its SLR so that each SLR stays under a
share (rho) of its resources:

  x[u,c]  one-hot: unit u uses candidate factor c
  y[u,k]  one-hot: unit u is placed in SLR k
  min     max_u latency_u(c)            (the slowest unit; or sum_u, see objective)
          then, within 1% of that latency, the smallest area,
          then, within 1% of that area, the smallest cut
  s.t.    sum_u area_u^r(c) * [u in k] <= rho_k * capacity_k^r   for all k, r
          grouping (shared on-chip buffers), anchors

Area and latency of a unit at a factor are not modelled: they are read from
graphs exported by the compiler (scalehls-export-dataflow-graph) with that
unit compiled at that factor. A compile takes seconds, so `floorplan.py sweep`
compiles the design once per factor with all units at that factor, and the
candidates of a unit are exactly the factors it was compiled with. Units that
share an external buffer influence each other slightly and HIDA's output
varies a little from run to run, so the chosen mix is compiled again and
compared with the budget by `floorplan.py budget-check`.
"""
from collections import defaultdict

import ilp

RES = ilp.RES


def build_tables(samples):
    """samples: unit -> [(factor, areas dict, latency)].
    Returns unit -> {factor: (areas, latency)}; repeated factors are averaged."""
    tables = {}
    for u, pts in samples.items():
        by = defaultdict(list)
        for f, areas, lat in pts:
            if f and f > 0: by[int(f)].append((areas, lat))
        if not by: continue
        tables[u] = {}
        for f, rows in by.items():
            areas = {r: sum(a.get(r, 0) for a, _ in rows) / len(rows) for r in RES}
            tables[u][f] = (areas, sum(l for _, l in rows) / len(rows))
    return tables


def area_at(tables, u, f):
    return tables[u][f][0]


def latency_at(tables, u, f):
    return tables[u][f][1]


def solve_budget(units, tables, edges, K, caps, *, fixed=None, objective='max', anchors=None,
                 groups=None, assignment=None, tolerance=0.01, time_limit=60.0, workers=4):
    """caps: list (per SLR) of {res: cap}. fixed: unit -> areas of units without
    a table (e.g. the memory pseudo-node). assignment: unit -> SLR to keep the
    placement fixed. tolerance: slack of each later objective; None stops after
    the latency objective. Returns (factors, assignment, info) or (None, None, info)."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    tun = [u for u in units if u in tables]            # units with a tunable factor
    fixed = fixed or {}
    y = {(u, k): m.NewBoolVar(f'y_{u}_{k}') for u in units for k in range(K)}
    p = {}
    for u in units:
        m.AddExactlyOne(y[u, k] for k in range(K))
        p[u] = m.NewIntVar(0, K - 1, f'p_{u}')
        m.Add(p[u] == sum(k * y[u, k] for k in range(K)))
    for src in (anchors or {}, assignment or {}):
        for u, k in src.items():
            if (u, k) in y: m.Add(y[u, k] == 1)
    for a, b in (groups or []):
        if a in p and b in p: m.Add(p[a] == p[b])
    x, z = {}, {}
    for u in tun:
        cs = sorted(tables[u])
        for c in cs: x[u, c] = m.NewBoolVar(f'x_{u}_{c}')
        m.AddExactlyOne(x[u, c] for c in cs)
        for c in cs:
            for k in range(K): z[u, c, k] = m.NewBoolVar(f'z_{u}_{c}_{k}')
            m.Add(sum(z[u, c, k] for k in range(K)) == x[u, c])
        for k in range(K):
            m.Add(sum(z[u, c, k] for c in cs) == y[u, k])
    for k in range(K):
        for r in RES:
            cap = caps[k].get(r, 0)
            if cap <= 0: continue
            terms = [int(round(area_at(tables, u, c)[r])) * z[u, c, k] for u in tun for c in tables[u]]
            terms += [int(fixed[u].get(r, 0)) * y[u, k] for u in units if u in fixed]
            m.Add(sum(terms) <= int(cap))
    # Latency, normalized so that the slowest configuration is about 1e6.
    worst = sum(max(l for _, l in tables[u].values()) for u in tun) or 1.0
    scale = 1e6 / worst
    lat = {u: sum(int(round(latency_at(tables, u, c) * scale)) * x[u, c] for c in tables[u]) for u in tun}
    if objective == 'max':
        goal = m.NewIntVar(0, int(2e6), 'max_latency')
        for u in tun: m.Add(goal >= lat[u])
    else:
        goal = sum(lat.values())
    # Area in 1/10000 of the summed capacity, so that resources are comparable.
    total = {r: sum(c.get(r, 0) for c in caps) for r in RES}
    area = sum(int(round(1e4 * area_at(tables, u, c)[r] / total[r])) * x[u, c]
               for u in tun for c in tables[u] for r in RES if total[r] > 0)
    cut = []
    for i, e in enumerate(edges):
        a, b = e['src'], e['dst']
        if a not in p or b not in p or a == b: continue
        w = int(round(10 * e['width'] * ilp.KIND_W.get(e['kind'], 1.0)))
        if not w: continue
        d = m.NewIntVar(0, K - 1, f'd{i}')
        m.Add(d >= p[a] - p[b]); m.Add(d >= p[b] - p[a])
        cut.append(w * d)

    def run(expr, hint):
        m.Minimize(expr)
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = time_limit
        s.parameters.num_workers = workers
        if hint is not None:
            m.ClearHints()
            for v in list(x.values()) + list(y.values()): m.AddHint(v, hint.Value(v))
        st = s.Solve(m)
        return (s if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None), s.StatusName(st), s.WallTime()

    s, status, seconds = run(goal, None)
    info = dict(status=status, seconds=round(seconds, 2))
    if s is None: return None, None, info
    if tolerance is not None:
        for expr in (area, sum(cut) if cut else None):
            if expr is None: continue
            prev = goal if expr is area else area
            m.Add(prev <= int(s.Value(prev) * (1 + tolerance)) + 1)
            s2, _, sec = run(expr, s)
            info['seconds'] = round(info['seconds'] + sec, 2)
            if s2 is None: break
            s = s2
    factors = {u: next(c for c in tables[u] if s.Value(x[u, c])) for u in tun}
    return factors, {u: s.Value(p[u]) for u in units}, info


def latency(tables, factors):
    """(sum, max) of the unit latencies."""
    vals = [latency_at(tables, u, f) for u, f in factors.items()]
    return sum(vals), max(vals)


def usage(tables, factors, assign, K, fixed=None):
    per = [defaultdict(float) for _ in range(K)]
    for u, f in factors.items():
        for r, v in area_at(tables, u, f).items(): per[assign[u]][r] += v
    for u, a in (fixed or {}).items():
        if u in assign:
            for r in RES: per[assign[u]][r] += a.get(r, 0)
    return [{r: int(round(d[r])) for r in RES} for d in per]


def fits(units, config, edges, K, caps, **kw):
    """Can a fixed configuration (unit -> (areas, latency)) be placed within the
    budget? Returns the assignment or None."""
    tables = {u: {1: v} for u, v in config.items()}
    kw = dict(kw, tolerance=None)
    factors, assign, _ = solve_budget(units, tables, edges, K, caps, **kw)
    return assign if factors is not None else None
