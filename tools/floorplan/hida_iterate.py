#!/usr/bin/env python3
"""Compiler-driven iterative floorplanning loop.

    iteration k:
      1. scalehls-opt -scalehls-physical-floorplan  (feedback of k-1 if k > 0)
            -> floorplan Tcl, assignment, next action (RTL passes, slot mode)
      2. netlist for the chosen RTL passes (cached per pass set):
            HLS RTL -> pipeline_ctrl.py (if any pass) -> Vivado OOC synthesis
      3. Vivado place (run_pnr.sh) with the floorplan, inside the strip
      4. make_feedback.py -> feedback k (sizes, connectivity, critical pairs,
         path classes, previous assignment)
    plus, once per RTL pass set, the same netlist placed without floorplan
    (Vivado-only baseline of identical RTL).

The decisions (which RTL pass, slot refinement, stop) are taken by the
compiler pass; this script only runs the tools and records iterations.csv.

usage: hida_iterate.py --mlir r18_t4_u16.mlir --base t4_u16 --work hida_t4_u16
"""
import argparse, csv, json, os, re, shutil, subprocess, sys

ROOT = '/mnt/d/fp_exp'
HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.environ.get('SCALEHLS_OPT', '/home/hainal/projects/thesis/build-fp/bin/scalehls-opt')
DEVICE = '/mnt/d/pr4_eval/graph/clock_regions.csv'


def sh(cmd, check=True):
    print('+', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f'failed ({r.returncode}): {cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    return r.stdout


def summary(out_dir):
    kv = dict(re.findall(r'(\w+)=(\S+)', open(os.path.join(out_dir, 'summary.txt')).read()))
    return float(kv['place_wns']), float(kv['place_tns'])


def netlist(base, work, passes):
    """Synthesized netlist of the base HLS RTL with the given RTL passes."""
    cfg = f'{work}/rtl_' + ('_'.join(passes) or 'orig')
    d = os.path.join(ROOT, cfg)
    if os.path.exists(os.path.join(d, 'post_synth.dcp')):
        return cfg
    vsrc = os.path.join(ROOT, base, 'hls_prj/sol/syn/verilog')
    vdst = os.path.join(d, 'hls_prj/sol/syn/verilog')
    if not passes:  # unmodified RTL: reuse the base synthesis
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(ROOT, base, 'post_synth.dcp'), d)
        return cfg
    if os.path.exists(vdst):
        shutil.rmtree(vdst)
    shutil.copytree(vsrc, vdst)
    flags = ' --ungate-axi' if 'ungate_axi' in passes else ''
    sh(f'{sys.executable} {HERE}/pipeline_ctrl.py {vsrc}/forward.v {vdst}/forward.v{flags}')
    win = cfg.replace('/', '\\\\')
    sh(f'cmd.exe /c "cd /d D:\\\\fp_exp && D:\\\\Vivado\\\\2023.2\\\\bin\\\\vivado.bat -mode batch -nojournal '
       f'-log {win}\\\\synth.log -source synth_ooc.tcl -tclargs {cfg}" > {cfg}/synth_stdout.txt 2>&1', check=False)
    if not os.path.exists(os.path.join(d, 'post_synth.dcp')):
        sys.exit(f'synthesis of {cfg} failed')
    return cfg


def place(cfg, fp, strip):
    out = os.path.join(ROOT, cfg, f'out_{fp}_Default_place')
    if not os.path.exists(os.path.join(out, 'summary.txt')):
        sh(f'./run_pnr.sh {cfg} {fp} Default place {strip}', check=False)
    if not os.path.exists(os.path.join(out, 'summary.txt')):
        sys.exit(f'place run {out} failed')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlir', required=True, help='HIDA output IR (before emission)')
    ap.add_argument('--base', required=True, help='dir with HLS RTL, post_synth.dcp, hier_util.rpt, graph/nets.csv')
    ap.add_argument('--work', required=True)
    ap.add_argument('--strip', default='0-2')
    ap.add_argument('--max-iters', type=int, default=8)
    ap.add_argument('--min-gain', type=float, default=0.1, help='WNS gain (ns) that counts as progress')
    ap.add_argument('--min-tns-gain', type=float, default=0.1, help='relative TNS gain that counts as progress')
    ap.add_argument('--measure-first', action='store_true',
                    help='iteration 0 starts from the feedback of an unconstrained Vivado run '
                         'instead of the compiler estimates')
    ap.add_argument('--pass-opts', default='', help='extra scalehls-physical-floorplan options')
    a = ap.parse_args()
    W = os.path.join(ROOT, a.work)
    os.makedirs(W, exist_ok=True)
    log_path = os.path.join(W, 'iterations.csv')
    new = not os.path.exists(log_path)
    log = open(log_path, 'a', newline='')
    w = csv.writer(log)
    if new:
        w.writerow(['iter', 'rtl_passes', 'slots', 'action', 'moved', 'slr_cut_weight', 'wns', 'tns',
                    'baseline_wns', 'baseline_tns', 'path_classes'])

    fb, best, best_tns, stall, baselines = None, None, None, 0, {}
    prev_passes = []
    if a.measure_first:
        # Vivado without floorplan on the unmodified RTL: measured sizes,
        # connectivity and path classes before any floorplan decision
        cfg = netlist(a.base, a.work, [])
        out = place(cfg, 'none', a.strip)
        baselines[cfg] = summary(out)
        empty = os.path.join(W, 'it_measure.json')
        json.dump(dict(assignment={}, slot_mode='slr'), open(empty, 'w'))
        fb = f'{a.work}/fb_measure.json'
        sh(f'{sys.executable} {HERE}/make_feedback.py --hier-util {a.base}/hier_util.rpt '
           f'--nets {a.base}/graph/nets.csv --rtl {a.base}/hls_prj/sol/syn/verilog '
           f'--run {out} --prev {empty} --applied= --out {fb}')
        cls = json.load(open(os.path.join(ROOT, fb)))['path_classes']
        b = baselines[cfg]
        w.writerow(['measure', '', '', 'no floorplan', 0, '', b[0], b[1], b[0], b[1],
                    ' '.join(f'{c}:{n}' for c, n in cls.items() if n)])
        log.flush()
        print(f'ITER measure wns={b[0]} tns={b[1]} classes={cls}', flush=True)
    for k in range(a.max_iters):
        js, tcl = f'{a.work}/it{k}.json', f'{a.work}/it{k}.tcl'
        opts = f'device={DEVICE} strip={a.strip} out-tcl={tcl} out-json={js}'
        if fb:
            opts += f' feedback={fb}'
        if a.pass_opts:
            opts += ' ' + a.pass_opts
        sh(f'{OPT} {a.mlir} -scalehls-physical-floorplan="{opts}" -o {a.work}/it{k}.mlir')
        plan = json.load(open(os.path.join(ROOT, js)))
        if plan['action'] == 'stop':
            print(f'ITER {k}: compiler pass says stop (worst paths are inside units)', flush=True)
            break
        passes = plan['rtl_passes']
        cfg = netlist(a.base, a.work, passes)
        if cfg not in baselines:
            baselines[cfg] = summary(place(cfg, 'none', a.strip))
        shutil.copy(os.path.join(ROOT, tcl), os.path.join(ROOT, cfg, f'fp_hida{k}.tcl'))
        out = place(cfg, f'hida{k}', a.strip)
        wns, tns = summary(out)
        fbk = f'{a.work}/fb{k}.json'
        cls = json.loads(sh(f'{sys.executable} {HERE}/make_feedback.py --hier-util {a.base}/hier_util.rpt '
                            f'--nets {a.base}/graph/nets.csv --rtl {a.base}/hls_prj/sol/syn/verilog '
                            f'--run {out} --prev {js} --applied="{",".join(passes)}" --out {fbk} '
                            f'&& python3 -c "import json;print(json.dumps(json.load(open(\'{fbk}\'))[\'path_classes\']))"'
                            ).strip().splitlines()[-1])
        b = baselines[cfg]
        w.writerow([k, '+'.join(passes), plan['slot_mode'], plan['action'], plan['moved'],
                    plan['slr_cut_weight'], wns, tns, b[0], b[1],
                    ' '.join(f'{c}:{n}' for c, n in cls.items() if n)])
        log.flush()
        print(f'ITER {k} {plan["action"]} rtl={passes} slots={plan["slot_mode"]} moved={plan["moved"]} '
              f'wns={wns} tns={tns} baseline={b} classes={cls}', flush=True)
        rtl_changed = passes != prev_passes
        progress = best is None or wns - best >= a.min_gain or \
            (best_tns < 0 and (tns - best_tns) / -best_tns >= a.min_tns_gain)
        best = wns if best is None else max(best, wns)
        best_tns = tns if best_tns is None else max(best_tns, tns)
        stall = 0 if (progress or rtl_changed) else stall + 1
        prev_passes = passes
        fb = fbk
        if stall >= 2:
            print(f'ITER {k}: converged (no WNS gain >= {a.min_gain} ns or TNS gain >= '
                  f'{a.min_tns_gain:.0%} twice)', flush=True)
            break


if __name__ == '__main__':
    main()
