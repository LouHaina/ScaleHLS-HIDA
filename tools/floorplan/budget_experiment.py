#!/usr/bin/env python3
"""Compiler-level experiment for the per-SLR resource budget.

For every tile size: compile the model at several unit factors and at several
values of the plain loop-unroll-factor knob, then for every budget level
(share of each SLR) choose per-unit factors with `floorplan.py budget`,
compile the chosen design again several times, check it against the budget
and compare its latency with the best plain-knob design that fits the same
budget. Everything is measured with HIDA's QoR estimate; no HLS or Vivado run
is involved.

  budget_experiment.py --mlir samples/pytorch/resnet18/resnet18.mlir \
      --device tools/floorplan/devices/u280.json --out-dir runs/budget

Needs scalehls-opt on PATH. Writes <out-dir>/results.json and prints a table.
"""
import argparse, json, os, statistics, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def compile_graph(a, tile, extra, graph):
    """Run the HIDA pipeline and export the graph. Returns the graph path or None."""
    if os.path.exists(graph) and not a.force: return graph
    cmd = (f'scalehls-opt {a.mlir} -hida-pytorch-pipeline="top-func={a.top} loop-tile-size={tile} {extra} '
           f'floorplan-graph-output={graph}" -o /dev/null')
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=a.timeout)
    except subprocess.TimeoutExpired:
        print(f'  timeout: tile {tile} {extra}', flush=True); return None
    if r.returncode or not os.path.exists(graph):
        print(f'  failed: tile {tile} {extra}: {r.stderr.strip()[-160:]}', flush=True)
        if os.path.exists(graph): os.remove(graph)
        return None
    return graph


def units_of(graph):
    g = json.load(open(graph))
    us = [n for n in g['nodes'] if n.get('is_unit') and not n['name'].startswith('__')]
    return sorted(us, key=lambda n: n['unit_index'])


def totals(graph):
    us = units_of(graph)
    return dict(latency_sum=sum(n['latency'] for n in us), latency_max=max(n['latency'] for n in us),
                dsp=sum(n['dsp'] for n in us))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mlir', required=True); ap.add_argument('--top', default='forward')
    ap.add_argument('--device', required=True); ap.add_argument('--out-dir', default='runs/budget')
    ap.add_argument('--tiles', default='2,4,8,16'); ap.add_argument('--factors', default='1,2,4,8,16,32,64,128')
    ap.add_argument('--knob', default='2,4,8,16,32,64,128,256'); ap.add_argument('--rhos', default='0.1,0.2,0.3,0.45,0.6')
    ap.add_argument('--objectives', default='sum,max'); ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 2); ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--force', action='store_true', help='recompile graphs that already exist')
    a = ap.parse_args()
    ints = lambda s: [int(v) for v in s.split(',') if v]
    results = []
    pool = ThreadPoolExecutor(a.jobs)
    for tile in ints(a.tiles):
        d = os.path.join(a.out_dir, f'tile{tile}'); os.makedirs(d, exist_ok=True)
        print(f'== tile {tile}: compiling', flush=True)
        knob_jobs = {F: pool.submit(compile_graph, a, tile, f'loop-unroll-factor={F}', f'{d}/knob_{F}.json') for F in ints(a.knob)}
        knob = {F: j.result() for F, j in knob_jobs.items() if j.result()}
        if not knob: print('  no knob compile succeeded; skipping'); continue
        n = len(units_of(next(iter(knob.values()))))
        jobs = {}
        for f in ints(a.factors):
            uf = f'{d}/all_{f}.factors.json'; json.dump({'unit_factors': [f] * n}, open(uf, 'w'))
            jobs[f] = pool.submit(compile_graph, a, tile, f'loop-unroll-factor=2 floorplan-unit-factors={uf}', f'{d}/all_{f}.json')
        unit = {f: j.result() for f, j in jobs.items() if j.result()}
        if not unit: print('  no unit-factor compile succeeded; skipping'); continue
        for rho in [float(v) for v in a.rhos.split(',')]:
            for obj in a.objectives.split(','):
                tag = f'{d}/rho{rho}_{obj}'
                r = subprocess.run([sys.executable, f'{HERE}/floorplan.py', 'budget', *unit.values(), '--baseline', *knob.values(),
                                    '--device', a.device, '--rho', str(rho), '--objective', obj,
                                    '--compile', f'scalehls-opt {a.mlir} -hida-pytorch-pipeline="top-func={a.top} loop-tile-size={tile} '
                                                 '{extra} floorplan-graph-output={graph}" -o /dev/null',
                                    '--out', f'{tag}.assignment.json', '--out-factors', f'{tag}.factors.json'],
                                   capture_output=True, text=True)
                row = dict(tile=tile, rho=rho, objective=obj, units=n)
                if r.returncode:
                    row['status'] = 'no factors fit' if r.returncode == 2 else 'failed: ' + r.stderr.strip()[-80:]
                    results.append(row); continue
                sol = json.load(open(f'{tag}.factors.json'))
                key = 'latency_max' if obj == 'max' else 'latency_sum'
                fit = [b for b in sol['baselines'] if b['fits']]
                best = min(fit, key=lambda b: b[key]) if fit else None
                ufit = [b for b in sol.get('uniform', []) if b['fits']]
                ubest = min(ufit, key=lambda b: b[key]) if ufit else None
                # Recompile the chosen design several times and check each one.
                finals = [pool.submit(compile_graph, a, tile, f'loop-unroll-factor=2 floorplan-unit-factors={tag}.factors.json',
                                      f'{tag}.final{i}.json') for i in range(a.repeat)]
                finals = [j.result() for j in finals if j.result()]
                checks = [subprocess.run([sys.executable, f'{HERE}/floorplan.py', 'budget-check', g, f'{tag}.assignment.json',
                                          '--device', a.device, '--rho', str(rho)], capture_output=True).returncode == 0 for g in finals]
                meas = [totals(g) for g in finals]
                row.update(status='ok', rounds=sol.get('rounds', 0), factors=sol['unit_factors'], predicted=sol['predicted_' + key],
                           measured=[m[key] for m in meas], dsp=[m['dsp'] for m in meas],
                           checks_passed=sum(checks), checks=len(checks),
                           knob=best['graph'] if best else None, knob_latency=best[key] if best else None,
                           knob_dsp=best['dsp'] if best else None,
                           same=ubest['graph'] if ubest else None, same_latency=ubest[key] if ubest else None)
                results.append(row)
        json.dump(results, open(os.path.join(a.out_dir, 'results.json'), 'w'), indent=1)
    print('\n| tile | share per SLR | objective | best knob that fits | knob latency | best same factor that fits | its latency | per-unit latency (median) | vs knob | vs same factor | spread of repeats | expected vs measured | extra rounds | budget check |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in results:
        if r['status'] != 'ok' or not r['measured']:
            print(f"| {r['tile']} | {r['rho']} | {r['objective']} | - | - | - | - | {r['status']} | - | - | - | - | - | - |"); continue
        med = statistics.median(r['measured']); spread = (max(r['measured']) - min(r['measured'])) / med
        knob = r['knob'].replace('knob_', '').replace('.json', '') if r['knob'] else 'none fits'
        kl = f"{r['knob_latency']:.3g}" if r['knob'] else '-'
        ratio = f"{r['knob_latency'] / med:.2f}x" if r['knob'] else '-'
        same = r['same'].replace('all_', '').replace('.json', '') if r.get('same') else 'none fits'
        sl = f"{r['same_latency']:.3g}" if r.get('same') else '-'
        sratio = f"{r['same_latency'] / med:.2f}x" if r.get('same') else '-'
        print(f"| {r['tile']} | {r['rho']} | {r['objective']} | {knob} | {kl} | {same} | {sl} | {med:.3g} | {ratio} | {sratio} | {spread:.1%} | "
              f"{(med - r['predicted']) / r['predicted']:+.1%} | {r['rounds']} | {r['checks_passed']}/{r['checks']} |")


if __name__ == '__main__':
    main()
