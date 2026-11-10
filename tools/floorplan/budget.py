"""Per-SLR resource budget: choose the unroll factor of every unit.

Plain HIDA has one knob, loop-unroll-factor, for the whole design. Here every
unit (top-level dataflow node) gets its own factor, chosen together with its
SLR so that each SLR stays under a share (rho) of its resources:

  x[u,c]  one-hot: unit u uses candidate factor c
  y[u,k]  one-hot: unit u is placed in SLR k
  min     sum_u latency_u(c)            (or max_u, see objective);
          then, within 1% of that latency, the smallest cut
  s.t.    sum_u area_u^r(c) * [u in k] <= rho_k * capacity_k^r   for all k, r
          grouping (shared on-chip buffers), anchors

Area and latency of a unit as a function of its factor are fitted from one or
more exported graphs (scalehls-export-dataflow-graph), each compiled at a
different factor:
  area^r(f)  = a + b*f      (one point: proportional for lut/ff/dsp,
                             constant for bram/uram)
  latency(f) = c + w/f      (one point: c = 0)
The fit is a prediction. After recompiling with the chosen factors, export the
graph again and run `floorplan.py budget-check`; pass that graph as a further
point to the next `budget` call to refine the fit.
"""
from collections import defaultdict

import ilp

RES = ilp.RES
PROPORTIONAL = ('lut', 'ff', 'dsp')


def candidate_factors(lo, hi):
    """1, 2, 3, 4, 6, 8, 12, 16, ... within [lo, hi]."""
    out, p = set(), 1
    while p <= hi:
        for v in (p, p + p // 2):
            if lo <= v <= hi: out.add(v)
        p *= 2
    return sorted(out)


def _fit_line(points, through_origin):
    """Least-squares y = a + b*x with a, b >= 0. points: [(x, y)]."""
    xs = sorted({x for x, _ in points})
    if len(xs) < 2:
        x, y = points[0]
        return (0.0, y / x) if through_origin and x > 0 else (float(y), 0.0)
    n = len(points); sx = sum(x for x, _ in points); sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points); sxy = sum(x * y for x, y in points)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    if b < 0: return sy / n, 0.0
    if a < 0: return 0.0, sxy / sxx
    return a, b


def fit_models(samples):
    """samples: unit -> [(factor, areas dict, latency)]. Returns unit -> model."""
    models, no_latency = {}, 0
    for u, pts in samples.items():
        pts = [p for p in pts if p[0] and p[0] > 0]
        if not pts: continue
        m = {'points': sorted({p[0] for p in pts}), 'area': {}}
        for r in RES:
            m['area'][r] = _fit_line([(f, a.get(r, 0)) for f, a, _ in pts], r in PROPORTIONAL)
        lat = [(1.0 / f, l) for f, _, l in pts if l and l > 0]
        if lat:
            m['lat'] = _fit_line(lat, True)          # (c, w): latency = c + w/f
            m['lat_measured'] = True
        else:
            # No estimate: HIDA gives a unit the factor 1 + F*share, so the
            # factor minus one is proportional to the unit's share of the work.
            f0 = pts[0][0]
            m['lat'] = (0.0, float(max(f0 - 1, 1)))
            m['lat_measured'] = False; no_latency += 1
        models[u] = m
    return models, no_latency


def area_at(model, f):
    return {r: max(0.0, a + b * f) for r, (a, b) in model['area'].items()}


def latency_at(model, f):
    c, w = model['lat']
    return c + w / f


def solve_budget(units, models, edges, K, caps, *, fixed=None, candidates=None, objective='sum',
                 anchors=None, groups=None, assignment=None, cut_tolerance=0.01, time_limit=60.0, workers=4):
    """caps: list (per SLR) of {res: cap}. fixed: unit -> SLR-independent areas
    of units without a model (e.g. the memory pseudo-node). candidates:
    unit -> [factors]. assignment: unit -> SLR to keep the placement fixed.
    Returns (factors, assignment, info) or (None, None, info)."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    tun = [u for u in units if u in models]            # units with a tunable factor
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
        cs = candidates[u]
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
            terms = [int(round(area_at(models[u], c)[r])) * z[u, c, k] for u in tun for c in candidates[u]]
            terms += [int(fixed[u].get(r, 0)) * y[u, k] for u in units if u in fixed]
            m.Add(sum(terms) <= int(cap))
    # Latency, normalized so that the slowest candidate configuration is ~1e6.
    worst = sum(max(latency_at(models[u], c) for c in candidates[u]) for u in tun) or 1.0
    scale = 1e6 / worst
    lat = {u: sum(int(round(latency_at(models[u], c) * scale)) * x[u, c] for c in candidates[u]) for u in tun}
    if objective == 'max':
        top = m.NewIntVar(0, int(2e6), 'max_latency')
        for u in tun: m.Add(top >= lat[u])
        goal = top
    else:
        goal = sum(lat.values())
    cut = []
    for i, e in enumerate(edges):
        a, b = e['src'], e['dst']
        if a not in p or b not in p or a == b: continue
        w = int(round(10 * e['width'] * ilp.KIND_W.get(e['kind'], 1.0)))
        if not w: continue
        d = m.NewIntVar(0, K - 1, f'd{i}')
        m.Add(d >= p[a] - p[b]); m.Add(d >= p[b] - p[a])
        cut.append(w * d)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    # Phase 1: best latency. Phase 2: among solutions within cut_tolerance of
    # it, the placement with the smallest cut.
    m.Minimize(goal)
    st = s.Solve(m)
    info = dict(status=s.StatusName(st), seconds=round(s.WallTime(), 2))
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, info
    if cut and cut_tolerance is not None:
        for v in list(x.values()) + list(y.values()): m.AddHint(v, s.Value(v))
        m.Add(goal <= int(s.Value(goal) * (1 + cut_tolerance)) + 1)
        m.Minimize(sum(cut))
        s2 = cp_model.CpSolver()
        s2.parameters.max_time_in_seconds = time_limit
        s2.parameters.num_workers = workers
        if s2.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            info['seconds'] = round(info['seconds'] + s2.WallTime(), 2); s = s2
    factors = {u: next(c for c in candidates[u] if s.Value(x[u, c])) for u in tun}
    return factors, {u: s.Value(p[u]) for u in units}, info


def total_latency(models, factors, objective='sum'):
    vals = [latency_at(models[u], f) for u, f in factors.items()]
    return max(vals) if objective == 'max' else sum(vals)


def usage(models, factors, assign, K, fixed=None):
    per = [defaultdict(float) for _ in range(K)]
    for u, f in factors.items():
        for r, v in area_at(models[u], f).items(): per[assign[u]][r] += v
    for u, a in (fixed or {}).items():
        if u in assign:
            for r in RES: per[assign[u]][r] += a.get(r, 0)
    return [{r: int(round(d[r])) for r in RES} for d in per]


def best_uniform(units, models, edges, K, caps, base, scales, **kw):
    """Largest common scale of the base factors that still fits: what the single
    HIDA knob can reach under the same per-SLR budget. Returns
    (scale, factors, assignment) or (None, None, None)."""
    best = (None, None, None)
    for s in sorted(scales):
        factors = {u: max(1, int(round(base[u] * s))) for u in base}
        cands = {u: [f] for u, f in factors.items()}
        f, asg, _ = solve_budget(units, models, edges, K, caps, candidates=cands, cut_tolerance=None, **kw)
        if f is None: break            # area grows with the scale: larger ones do not fit either
        best = (s, f, asg)
    return best
