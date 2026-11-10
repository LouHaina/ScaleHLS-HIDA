#!/usr/bin/env python3
"""Checks for the per-SLR budget solver. Run: python3 tools/floorplan/test_budget.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as bg  # noqa: E402


def models_for(spec):
    """spec: unit -> (base factor, dsp at base, latency at base); two points each."""
    samples = {}
    for u, (f, dsp, lat) in spec.items():
        samples[u] = [(f, {'dsp': dsp, 'bram': 10}, lat), (2 * f, {'dsp': 2 * dsp, 'bram': 10}, lat / 2)]
    models, missing = bg.fit_models(samples)
    assert missing == 0
    return models


def test_fit():
    m = models_for({'a': (4, 100, 800.0)})['a']
    assert abs(bg.area_at(m, 8)['dsp'] - 200) < 1e-6 and abs(bg.area_at(m, 16)['dsp'] - 400) < 1e-6
    assert abs(bg.area_at(m, 16)['bram'] - 10) < 1e-6          # constant across the two points
    assert abs(bg.latency_at(m, 16) - 200) < 1e-6
    # One point: proportional for dsp, constant for bram, latency ~ 1/f.
    one, _ = bg.fit_models({'a': [(4, {'dsp': 100, 'bram': 10}, 800.0)]})
    assert abs(bg.area_at(one['a'], 8)['dsp'] - 200) < 1e-6 and abs(bg.area_at(one['a'], 8)['bram'] - 10) < 1e-6
    assert abs(bg.latency_at(one['a'], 8) - 400) < 1e-6
    # No latency estimate: the base factor stands in for the share of the work.
    none, missing = bg.fit_models({'a': [(5, {'dsp': 100}, 0)]})
    assert missing == 1 and not none['a']['lat_measured']


def test_budget_respects_each_slr():
    spec = {f'u{i}': (4, 100, 1000.0) for i in range(6)}
    models = models_for(spec); units = list(spec)
    caps = [{'dsp': 400}, {'dsp': 200}]                        # second SLR is half as large
    cands = {u: bg.candidate_factors(1, 32) for u in units}
    factors, assign, info = bg.solve_budget(units, models, [], 2, caps, candidates=cands)
    assert factors is not None, info
    use = bg.usage(models, factors, assign, 2)
    assert use[0]['dsp'] <= 400 and use[1]['dsp'] <= 200


def test_heavy_unit_gets_the_resources():
    # One unit holds most of the latency; a common scale cannot favour it.
    spec = {'heavy': (2, 100, 10000.0), 'l1': (2, 100, 100.0), 'l2': (2, 100, 100.0)}
    models = models_for(spec); units = list(spec)
    caps = [{'dsp': 600}]
    cands = {u: bg.candidate_factors(1, 32) for u in units}
    factors, assign, _ = bg.solve_budget(units, models, [], 1, caps, candidates=cands)
    assert factors['heavy'] > factors['l1'] and factors['heavy'] > factors['l2']
    base = {u: spec[u][0] for u in units}
    scale, uniform, _ = bg.best_uniform(units, models, [], 1, caps, base, [0.5, 1, 2, 4])
    assert scale == 2                                          # 3 units * 200 dsp = 600
    assert bg.total_latency(models, factors) < bg.total_latency(models, uniform)


def test_infeasible_and_fixed_assignment():
    spec = {'a': (4, 400, 100.0), 'b': (4, 400, 100.0)}
    models = models_for(spec); units = list(spec)
    cands = {u: [4] for u in units}
    assert bg.solve_budget(units, models, [], 1, [{'dsp': 500}], candidates=cands)[0] is None
    factors, assign, _ = bg.solve_budget(units, models, [], 2, [{'dsp': 500}, {'dsp': 500}], candidates=cands,
                                         assignment={'a': 1})
    assert assign['a'] == 1 and assign['b'] == 0


def test_cut_is_minimized_within_tolerance():
    spec = {u: (4, 100, 1000.0) for u in 'abcd'}
    models = models_for(spec); units = list(spec)
    edges = [dict(src='a', dst='b', width=64, kind='stream'), dict(src='c', dst='d', width=64, kind='stream')]
    cands = {u: [4] for u in units}
    _, assign, _ = bg.solve_budget(units, models, edges, 2, [{'dsp': 200}, {'dsp': 200}], candidates=cands)
    assert assign['a'] == assign['b'] and assign['c'] == assign['d']


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok', t.__name__)
    print(f'{len(tests)} passed')
