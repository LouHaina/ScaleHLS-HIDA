#!/usr/bin/env python3
"""Summarize run_pnr.tcl output dirs: WNS/TNS per stage, Fmax, and what the
worst paths are made of (SLR crossings, logic vs route delay, unit pairs).

usage: summarize_runs.py <cfg_dir> [run_dir_glob]"""
import csv, glob, os, re, sys
from collections import Counter

cfg = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else 'out_*'
PERIOD = 3.33


def unit(pin):
    h = pin.split('/')[0]
    return 'top' if h == pin else h


for d in sorted(glob.glob(os.path.join(cfg, pat))):
    s = os.path.join(d, 'summary.txt')
    if not os.path.exists(s):
        print(f'{os.path.basename(d):40s} (no summary)')
        continue
    kv = dict(re.findall(r'(\w+)=(\S+)', open(s).read()))
    line = f"{os.path.basename(d):40s}"
    for st in ('place', 'route'):
        if f'{st}_wns' in kv:
            w = float(kv[f'{st}_wns'])
            line += f"  {st}: WNS {w:+.3f} TNS {kv.get(f'{st}_tns','?'):>10s} Fmax {1000/(PERIOD-w):6.1f}"
    line += f"  t={kv.get('total_s', kv.get('place_s'))}s"
    print(line)
    for st in ('route', 'place'):
        p = os.path.join(d, f'paths_{st}.csv')
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p)))[:100]
            if not rows:
                break
            xs = sum(int(r['inter_slr']) for r in rows)
            lg = sum(float(r['logic']) for r in rows) / len(rows)
            rt = sum(float(r['route']) for r in rows) / len(rows)
            lv = sum(int(r['levels']) for r in rows) / len(rows)
            pairs = Counter(f"{unit(r['start'])}->{unit(r['end'])}" for r in rows)
            print(f"    worst100 ({st}): inter-SLR {xs}%  logic {lg:.2f}ns route {rt:.2f}ns levels {lv:.1f}")
            for k, v in pairs.most_common(4):
                print(f"      {v:3d}  {k}")
            break
