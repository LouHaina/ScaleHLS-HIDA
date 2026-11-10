#!/usr/bin/env python3
"""Like-for-like evaluation of floorplan arms against the unconstrained baseline.

Every arm goes through the same Vivado script (run_vivado_iter.tcl), the same
clock period and the same list of placer directives; arms differ only in the
checkpoint and the constraint file. One P&R run is noise (the runs in data/
swing by several ns on near-identical constraints), so each arm is run once
per directive and compared by its median.

  eval_sweep.py run        eval_config.json   run the missing (arm, directive) pairs
  eval_sweep.py summarize  eval_config.json   table + summary.csv from finished runs
  eval_sweep.py summarize --dirs data/run_0_2 data/run_0_2_1 --period 8.0
                                              same table for arbitrary report directories

Config keys (see eval_config.example.json):
  design      label of the design point (use one config per size of a sweep)
  period      clock period in ns, shared by every arm
  out         output directory; runs land in <out>/<arm>/<directive>/
  directives  place_design directives, e.g. ["Default", "Explore", "ExtraTimingOpt"]
  vivado      command template with {dcp} {tcl} {period} {out} {directive}
  arms        name -> {"dcp": post-synthesis checkpoint, "tcl": constraints or "none"}

The usual four arms separate the two effects (AutoBridge / RapidStream style):
  baseline        plain HIDA checkpoint, no constraints
  pipeline_only   checkpoint compiled with the floorplan (deeper cut FIFOs), no constraints
  floorplan_only  plain HIDA checkpoint, floorplan constraints
  full            floorplan checkpoint, floorplan constraints

Fmax = 1000 / (period - WNS) from the post-route timing summary. A run that
fails placement or routing has no Fmax and is counted in "failed", never
averaged in as 0 MHz.
"""
import argparse, csv, json, os, re, statistics, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_reports  # noqa: E402


def run_dir(cfg, arm, directive):
    return os.path.join(cfg['out'], arm, directive)


def cmd_run(a):
    cfg = json.load(open(a.config))
    for arm, spec in cfg['arms'].items():
        for directive in cfg.get('directives', ['Default']):
            d = run_dir(cfg, arm, directive)
            if os.path.exists(os.path.join(d, 'status.txt')) and not a.force:
                print(f'skip {arm}/{directive} (finished)'); continue
            os.makedirs(d, exist_ok=True)
            cmd = cfg['vivado'].format(dcp=spec['dcp'], tcl=spec.get('tcl') or 'none', period=cfg['period'],
                                       out=d, directive=directive)
            print('+', cmd, flush=True)
            if a.dry_run: continue
            r = subprocess.run(cmd, shell=True)
            if r.returncode and not os.path.exists(os.path.join(d, 'status.txt')):
                open(os.path.join(d, 'status.txt'), 'w').write(f'tool_failed exit={r.returncode}\n')
    if not a.dry_run: cmd_summarize(a)


def read_run(d, period):
    """One report directory -> dict(status, wns, fmax, crossings, max_slr_lut_pct)."""
    rec = dict(dir=d, status='missing', wns=None, fmax=None, tns=None, crossings=None, max_slr_lut_pct=None)
    st = os.path.join(d, 'status.txt')
    timing = os.path.join(d, 'timing_summary.rpt')
    if os.path.exists(st):
        rec['status'] = open(st).read().split()[0]
    elif os.path.exists(timing):
        rec['status'] = 'done'           # older runs without a status file
    if rec['status'] != 'done': return rec
    route = os.path.join(d, 'route_status.rpt')
    if os.path.exists(route):
        m = re.search(r'# of nets with routing errors\.*\s*:\s*(\d+)', open(route, errors='replace').read())
        if m and int(m.group(1)) > 0:
            rec['status'] = 'route_failed'; return rec
    if not os.path.exists(timing):
        rec['status'] = 'missing'; return rec
    t = parse_reports.parse_timing_summary(timing)
    if 'wns' not in t:
        rec['status'] = 'no_timing'; return rec
    rec['wns'], rec['tns'] = t['wns'], t['tns']
    rec['fmax'] = round(1000.0 / (period - t['wns']), 1)
    util = os.path.join(d, 'utilization_slr.rpt')
    if os.path.exists(util):
        u = parse_reports.parse_utilization_slr(util)
        if u['sll']: rec['crossings'] = sum(v['used'] for v in u['sll'].values())
        pcts = [v.get('lut_pct') for v in u['per_slr'].values() if v.get('lut_pct') is not None]
        if pcts: rec['max_slr_lut_pct'] = max(pcts)
    return rec


def summarize(groups, period, design, out_csv=None):
    """groups: arm -> list of (label, dir)."""
    rows, table = [], []
    for arm, runs in groups.items():
        recs = []
        for label, d in runs:
            r = read_run(d, period); r.update(design=design, arm=arm, run=label); recs.append(r); rows.append(r)
        ok = [r for r in recs if r['status'] == 'done']
        failed = [r for r in recs if r['status'] not in ('done', 'missing')]
        f = [r['fmax'] for r in ok]; w = [r['wns'] for r in ok]
        x = [r['crossings'] for r in ok if r['crossings'] is not None]
        table.append(dict(arm=arm, runs=len(ok), failed=len(failed), missing=len(recs) - len(ok) - len(failed),
                          fmax_median=round(statistics.median(f), 1) if f else None,
                          fmax_min=min(f) if f else None, fmax_max=max(f) if f else None,
                          wns_median=round(statistics.median(w), 3) if w else None,
                          met=sum(1 for v in w if v >= 0),
                          crossings_median=int(statistics.median(x)) if x else None))
    base = next((t['fmax_median'] for t in table if t['arm'] == 'baseline'), None)
    print(f'design: {design}   period: {period} ns ({1000 / period:.0f} MHz target)')
    print('| arm | runs | failed | Fmax median (MHz) | Fmax min-max | vs baseline | WNS median (ns) | met | SLR crossings |')
    print('|---|---|---|---|---|---|---|---|---|')
    for t in table:
        if t['fmax_median'] is None:
            print(f"| {t['arm']} | 0 | {t['failed']} | - | - | - | - | - | - |"); continue
        rel = f"{(t['fmax_median'] / base - 1) * 100:+.1f}%" if base and t['arm'] != 'baseline' else '-'
        cross = t['crossings_median'] if t['crossings_median'] is not None else '-'
        print(f"| {t['arm']} | {t['runs']} | {t['failed']} | {t['fmax_median']} | {t['fmax_min']}-{t['fmax_max']} | {rel} | "
              f"{t['wns_median']} | {t['met']}/{t['runs']} | {cross} |")
    if any(t['runs'] and t['runs'] < 3 for t in table):
        print('note: fewer than 3 runs in an arm; differences below the min-max spread are not evidence.')
    if out_csv:
        cols = ['design', 'arm', 'run', 'status', 'wns', 'tns', 'fmax', 'crossings', 'max_slr_lut_pct', 'dir']
        with open(out_csv, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore'); wr.writeheader(); wr.writerows(rows)
        print(f'wrote {out_csv}')
    return table


def cmd_summarize(a):
    if getattr(a, 'dirs', None):
        if a.period is None: raise SystemExit('--dirs needs --period')
        groups = {os.path.basename(os.path.normpath(d)): [('run', d)] for d in a.dirs}
        return summarize(groups, a.period, 'ad hoc', a.csv)
    if not a.config: raise SystemExit('give a config file or --dirs')
    cfg = json.load(open(a.config))
    groups = {arm: [(dv, run_dir(cfg, arm, dv)) for dv in cfg.get('directives', ['Default'])] for arm in cfg['arms']}
    return summarize(groups, cfg['period'], cfg.get('design', '?'), os.path.join(cfg['out'], 'summary.csv'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('config'); r.add_argument('--force', action='store_true')
    r.add_argument('--dry-run', action='store_true'); r.set_defaults(fn=cmd_run)
    s = sub.add_parser('summarize'); s.add_argument('config', nargs='?'); s.add_argument('--dirs', nargs='+')
    s.add_argument('--period', type=float); s.add_argument('--csv'); s.set_defaults(fn=cmd_summarize)
    a = ap.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
