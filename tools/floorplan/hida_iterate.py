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
    ap.add_argument('--min-gain', type=float, default=0.02)
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

    fb, best, stall, baselines = None, None, 0, {}
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
                            f'--run {out} --prev {js} --applied {",".join(passes)} --out {fbk} '
                            f'&& python3 -c "import json;print(json.dumps(json.load(open(\'{fbk}\'))[\'path_classes\']))"'
                            ).strip().splitlines()[-1])
        b = baselines[cfg]
        w.writerow([k, '+'.join(passes), plan['slot_mode'], plan['action'], plan['moved'],
                    plan['slr_cut_weight'], wns, tns, b[0], b[1],
                    ' '.join(f'{c}:{n}' for c, n in cls.items() if n)])
        log.flush()
        print(f'ITER {k} {plan["action"]} rtl={passes} slots={plan["slot_mode"]} moved={plan["moved"]} '
              f'wns={wns} tns={tns} baseline={b} classes={cls}', flush=True)
        rtl_changed = k > 0 and passes != prev_passes
        gain = None if best is None else wns - best
        best = wns if best is None else max(best, wns)
        stall = 0 if (gain is None or gain >= a.min_gain or rtl_changed) else stall + 1
        prev_passes = passes
        fb = fbk
        if stall >= 2:
            print(f'ITER {k}: converged (WNS gain < {a.min_gain} ns twice)', flush=True)
            break


if __name__ == '__main__':
    main()
