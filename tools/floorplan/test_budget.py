#!/usr/bin/env python3
"""Checks for the per-SLR budget solver. Run: python3 tools/floorplan/test_budget.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as bg  # noqa: E402


def tables_for(spec, factors=(1, 2, 4, 8, 16)):
    """spec: unit -> (dsp at factor 1, latency at factor 1); dsp grows and
    latency shrinks in proportion to the factor."""
    samples = {u: [(f, {'dsp': dsp * f, 'bram': 10}, lat / f) for f in factors] for u, (dsp, lat) in spec.items()}
    return bg.build_tables(samples)


def test_tables_average_repeated_factors():
    t = bg.build_tables({'a': [(4, {'dsp': 100}, 800.0), (4, {'dsp': 120}, 900.0), (8, {'dsp': 200}, 400.0), (0, {'dsp': 1}, 1.0)]})
    assert sorted(t['a']) == [4, 8]
    assert bg.area_at(t, 'a', 4)['dsp'] == 110 and bg.latency_at(t, 'a', 4) == 850.0
    assert bg.area_at(t, 'a', 4)['bram'] == 0


def test_budget_respects_each_slr():
    spec = {f'u{i}': (25, 1000.0) for i in range(6)}
    tables = tables_for(spec); units = list(spec)
    caps = [{'dsp': 400}, {'dsp': 200}]                        # second SLR is half as large
    factors, assign, info = bg.solve_budget(units, tables, [], 2, caps)
    assert factors is not None, info
    use = bg.usage(tables, factors, assign, 2)
    assert use[0]['dsp'] <= 400 and use[1]['dsp'] <= 200
    assert max(factors.values()) > 1                           # and the budget is actually used


def test_heavy_unit_gets_the_resources():
    spec = {'heavy': (50, 10000.0), 'l1': (50, 100.0), 'l2': (50, 100.0)}
    tables = tables_for(spec); units = list(spec)
    factors, _, _ = bg.solve_budget(units, tables, [], 1, [{'dsp': 600}], objective='sum')
    assert factors['heavy'] > factors['l1'] and factors['heavy'] > factors['l2']
    same = {u: 4 for u in units}                               # 3 * 200 dsp: the largest common factor that fits
    assert bg.latency(tables, factors)[0] < bg.latency(tables, same)[0]


def test_factor_that_does_not_help_is_not_chosen():
    # Unrolling "flat" costs area and gains nothing; the area objective keeps it small.
    samples = {'flat': [(f, {'dsp': 10 * f}, 500.0) for f in (1, 2, 4, 8)],
               'good': [(f, {'dsp': 10 * f}, 800.0 / f) for f in (1, 2, 4, 8)]}
    tables = bg.build_tables(samples)
    factors, _, _ = bg.solve_budget(['flat', 'good'], tables, [], 1, [{'dsp': 1000}], objective='sum')
    assert factors == {'flat': 1, 'good': 8}


def test_max_objective_balances_units():
    spec = {'a': (10, 1600.0), 'b': (10, 400.0)}
    tables = tables_for(spec)
    factors, _, _ = bg.solve_budget(['a', 'b'], tables, [], 1, [{'dsp': 100}], objective='max')
    assert bg.latency(tables, factors)[1] <= 200.0             # a at 8 (200), b at 2 (200): 100 dsp


def test_infeasible_fixed_assignment_and_fits():
    spec = {'a': (400, 100.0), 'b': (400, 100.0)}
    tables = tables_for(spec, factors=(1,)); units = list(spec)
    assert bg.solve_budget(units, tables, [], 1, [{'dsp': 500}])[0] is None
    _, assign, _ = bg.solve_budget(units, tables, [], 2, [{'dsp': 500}, {'dsp': 500}], assignment={'a': 1})
    assert assign['a'] == 1 and assign['b'] == 0
    config = {u: ({'dsp': 400}, 100.0) for u in units}
    assert bg.fits(units, config, [], 1, [{'dsp': 500}]) is None
    assert bg.fits(units, config, [], 2, [{'dsp': 500}, {'dsp': 500}]) is not None


def test_cut_is_minimized_last():
    spec = {u: (100, 1000.0) for u in 'abcd'}
    tables = tables_for(spec, factors=(1,)); units = list(spec)
    edges = [dict(src='a', dst='b', width=64, kind='stream'), dict(src='c', dst='d', width=64, kind='stream')]
    _, assign, _ = bg.solve_budget(units, tables, edges, 2, [{'dsp': 200}, {'dsp': 200}])
    assert assign['a'] == assign['b'] and assign['c'] == assign['d']


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok', t.__name__)
    print(f'{len(tests)} passed')
