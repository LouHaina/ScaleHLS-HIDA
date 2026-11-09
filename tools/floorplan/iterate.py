#!/usr/bin/env python3
"""Iterative physical-feedback floorplanning loop (PR #4 design doc Sec. 8, extended).

Each iteration:
  1. place the current design + floorplan with Vivado (run_pnr.sh, place stage);
  2. classify the worst paths of that run;
  3. choose the next action from the dominant path class:
       ctrl_broadcast  -> RTL pass: register the top-level all-subcalls-done
                          condition (pipeline_ctrl.py), re-synthesize
       fsm_axi_gate    -> RTL pass: remove FSM gating of single-user m_axi
                          handshakes (pipeline_ctrl.py --ungate-axi), re-synthesize
       placement       -> add the worst paths' unit pairs as ILP feedback edges;
                          if they are long routes inside one SLR and the slots are
                          SLRs, refine the slots to half-SLR bands
       unit_internal   -> nothing a floorplan can change: stop
  4. re-solve the ILP (grid_floorplan.py, stability term against the previous
     assignment) and continue.
Whenever the RTL changes, the same RTL is also placed without floorplan, so each
iteration is compared against a Vivado-only baseline of identical RTL.
Stops when no action applies or WNS improves by < --min-gain twice in a row.

Every iteration is appended to <work>/iterations.csv.

usage: iterate.py --base t4_u16 --work it_t4_u16 [--strip 0-2] [--max-iters 8]
       (run from WSL; Vivado is driven through cmd.exe, paths under /mnt/d/fp_exp)
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, time
from collections import Counter

ROOT = '/mnt/d/fp_exp'
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CRS = '/mnt/d/pr4_eval/graph/clock_regions.csv'


def sh(cmd):
    print('+', cmd, flush=True)
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True).stdout


def unit(pin):
    h = pin.split('/')[0]
    return 'top' if h == pin else h


def classify(paths_csv, top_n=100):
    rows = list(csv.DictReader(open(paths_csv)))[:top_n]
    c = Counter()
    for r in rows:
        s, e = unit(r['start']), unit(r['end'])
        logic, route = float(r['logic']), float(r['route'])
        if re.search(r'ap_CS_fsm', r['start']) and re.match(r'axi_\d+_m_axi_U', e):
            c['fsm_axi_gate'] += 1
        elif s != e and s.startswith('grp_') and e.startswith('grp_') and \
                re.search(r'loop_dataflow_output_count|ap_sync_reg|ap_done_reg', r['end']):
            c['ctrl_broadcast'] += 1
        elif s == e and logic >= route:
            c['unit_internal'] += 1
        else:
            c['placement_xslr' if r['inter_slr'] == '1' else 'placement_local'] += 1
    return c, rows


def summary(out_dir):
    kv = dict(re.findall(r'(\w+)=(\S+)', open(os.path.join(out_dir, 'summary.txt')).read()))
    return float(kv['place_wns']), float(kv['place_tns'])


def make_rtl(base, cfg, passes):
    """Copy <base>'s HLS RTL into <cfg>, apply the RTL passes, synthesize."""
    d = os.path.join(ROOT, cfg)
    if os.path.exists(os.path.join(d, 'post_synth.dcp')):
        return
    vsrc = os.path.join(ROOT, base, 'hls_prj/sol/syn/verilog')
    vdst = os.path.join(d, 'hls_prj/sol/syn/verilog')
    if os.path.exists(vdst):
        shutil.rmtree(vdst)
    shutil.copytree(vsrc, vdst)
    if passes:
        args = ' --ungate-axi' if 'ungate_axi' in passes else ''
        sh(f'{PY} {HERE}/pipeline_ctrl.py {vsrc}/forward.v {vdst}/forward.v{args}')
    win = cfg.replace('/', '\\\\')
    sh(f'cmd.exe /c "cd /d D:\\\\fp_exp && D:\\\\Vivado\\\\2023.2\\\\bin\\\\vivado.bat -mode batch -nojournal '
       f'-log {win}\\\\synth.log -source synth_ooc.tcl -tclargs {cfg}" > {cfg}/synth_stdout.txt 2>&1')
    if not os.path.exists(os.path.join(d, 'post_synth.dcp')):
        sys.exit(f'synthesis of {cfg} failed')


def place(cfg, fp, strip):
    out = os.path.join(ROOT, cfg, f'out_{fp}_Default_place')
    if not os.path.exists(os.path.join(out, 'summary.txt')):
        sh(f'./run_pnr.sh {cfg} {fp} Default place {strip}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='workload dir with HLS RTL, hier_util.rpt, graph/nets.csv')
    ap.add_argument('--work', required=True, help='work dir (under /mnt/d/fp_exp) for this loop')
    ap.add_argument('--strip', default='0-2')
    ap.add_argument('--max-iters', type=int, default=8)
    ap.add_argument('--min-gain', type=float, default=0.02)
    ap.add_argument('--fb-weight', type=float, default=50)
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, a.work), exist_ok=True)
    log_path = os.path.join(ROOT, a.work, 'iterations.csv')
    new_log = not os.path.exists(log_path)
    log = open(log_path, 'a', newline='')
    w = csv.writer(log)
    if new_log:
        w.writerow(['iter', 'rtl_passes', 'slots', 'feedback_paths', 'moved_units', 'wns', 'tns',
                    'baseline_wns', 'baseline_tns', 'path_classes', 'next_action'])

    passes, mode, fb, prev_json = [], 'slr', [], None
    best, stall, baseline = None, 0, {}
    for k in range(a.max_iters):
        rtl = 'rtl_' + ('_'.join(passes) or 'orig')
        cfg = f'{a.work}/{rtl}'
        make_rtl(a.base, cfg, passes)
        if rtl not in baseline:
            baseline[rtl] = summary(place(cfg, 'none', a.strip))
        # floorplan for this iteration
        fbcsv = os.path.join(ROOT, a.work, f'fb_{k}.csv')
        with open(fbcsv, 'w') as f:
            f.write('slack,levels,logic,route,inter_slr,start,end\n')
            for p in fb:
                f.writelines(open(p).readlines()[1:101])
        pin = 'top={0},control_s_axi_U={0}'.format('SLR1' if mode == 'slr' else 'B6')
        jp = os.path.join(ROOT, cfg, f'it{k}.json')
        done = os.path.exists(os.path.join(ROOT, cfg, f'out_it{k}_Default_place', 'summary.txt'))
        if not done:
          sh(f'{PY} {HERE}/grid_floorplan.py --hier-util {a.base}/hier_util.rpt --nets {a.base}/graph/nets.csv '
           f'--rtl {a.base}/hls_prj/sol/syn/verilog '
           f'--crs {CRS} --slot-mode {mode} --strip {a.strip} --rho 0.75 --balance 2000 --pin {pin} '
           + (f'--feedback {fbcsv} --feedback-top {100 * len(fb)} --feedback-weight {a.fb_weight} ' if fb else '')
           + (f'--prev {prev_json} ' if prev_json else '')
           + f'--time 300 --out-tcl {cfg}/fp_it{k}.tcl --out-json {jp}')  # (skipped if this iteration already ran)
        asg = json.load(open(jp))['assignment']
        moved = 0 if not prev_json else sum(
            1 for n, s in asg.items() if json.load(open(prev_json))['assignment'].get(n) != s)
        out = place(cfg, f'it{k}', a.strip)
        wns, tns = summary(out)
        classes, _ = classify(os.path.join(out, 'paths_place.csv'))
        top_cls = classes.most_common(1)[0][0]
        # next action
        if top_cls == 'ctrl_broadcast' and 'ctrl' not in passes:
            action = 'rtl:ctrl'; passes = passes + ['ctrl']
        elif top_cls == 'fsm_axi_gate' and 'ungate_axi' not in passes:
            action = 'rtl:ungate_axi'; passes = sorted(set(passes + ['ctrl', 'ungate_axi']))
        elif top_cls.startswith('placement'):
            fb = fb + [os.path.join(out, 'paths_place.csv')]
            action = 'feedback'
            if top_cls == 'placement_local' and mode == 'slr':
                mode = 'band'; action = 'feedback+refine:band'; prev_json = None
        else:
            action = 'stop'
        gain = None if best is None else wns - best
        best = wns if best is None else max(best, wns)
        stall = stall + 1 if (gain is not None and gain < a.min_gain and not action.startswith('rtl')) else 0
        b = baseline[rtl]
        w.writerow([k, rtl[4:], mode,
                    len(fb), moved, wns, tns, b[0], b[1],
                    ' '.join(f'{c}:{n}' for c, n in classes.most_common()), action])
        log.flush()
        print(f'ITER {k} {cfg} wns={wns} tns={tns} baseline={b} classes={dict(classes)} -> {action}', flush=True)
        if action == 'stop' or stall >= 2:
            break
        if not action.endswith('band'):
            prev_json = jp  # keep the assignment stable unless the slot grid changed


if __name__ == '__main__':
    main()
