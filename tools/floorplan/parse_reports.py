#!/usr/bin/env python3
"""Parse Vivado reports of one P&R iteration into feedback.json.

Inputs (all produced by run_vivado_iter.tcl, or by the older run.tcl):
  timing_summary.rpt   -> WNS/TNS and the worst paths (source/destination cells)
  utilization_slr.rpt  -> per-SLR LUT/FF/BRAM/DSP/URAM and SLL usage per boundary
  congestion.rpt       -> "SLR Net Crossing Reporting" table (per-cell crossing counts)
  crossing_paths.csv   -> optional, from get_timing_paths -inter_slr (slack, start, end)

Every cell name is mapped to a HIDA node by the hierarchy prefix
(grp_<node>_fu_N or <node>_U0), so the result is expressed in compiler terms:
failing node pairs, crossing nets per node, and per-SLR utilization.

Usage: parse_reports.py <run_dir> [--out feedback.json]
"""
import argparse, json, os, re
from collections import defaultdict

NODE_RE = re.compile(r'(?:^|/)(?:grp_)?(\w+?_node\d+)(?:_\d+)?(?:_fu_\d+|_U0)(?:/|$)')

def cell_to_node(cell):
    m = NODE_RE.search(cell)
    return m.group(1) if m else None

def parse_timing_summary(path):
    txt = open(path, errors='replace').read()
    out = {}
    m = re.search(r'WNS\(ns\).*?\n.*?\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)', txt, re.S)
    if m:
        out.update(wns=float(m.group(1)), tns=float(m.group(2)), failing_endpoints=int(m.group(3)),
                   total_endpoints=int(m.group(4)), whs=float(m.group(5)))
    paths = []
    for pm in re.finditer(r'Slack \((VIOLATED|MET)\) :\s+(-?[\d.]+)ns\s+\(required time - arrival time\).*?Source:\s+(\S+).*?Destination:\s+(\S+).*?'
                          r'Data Path Delay:\s+([\d.]+)ns\s+\(logic ([\d.]+)ns.*?route ([\d.]+)ns', txt, re.S):
        src, dst = pm.group(3), pm.group(4)
        paths.append(dict(slack=float(pm.group(2)), source=src, destination=dst,
                          src_node=cell_to_node(src), dst_node=cell_to_node(dst),
                          delay=float(pm.group(5)), logic=float(pm.group(6)), route=float(pm.group(7))))
    out['worst_paths'] = paths
    return out

def parse_utilization_slr(path):
    txt = open(path, errors='replace').read()
    out = {'per_slr': {}, 'sll': {}}
    sec = re.search(r'SLR CLB Logic and Dedicated Block Utilization\n-+\n(.*?)\n\n', txt, re.S)
    if sec:
        hdr = re.search(r'\|\s*Site Type\s*\|(.*?)\n', sec.group(1)).group(1)
        cols = [c.strip() for c in hdr.split('|') if c.strip()]
        slrs = [c for c in cols if re.fullmatch(r'SLR\d', c)]
        want = {'CLB LUTs': 'lut', 'CLB Registers': 'ff', 'Block RAM Tile': 'bram', 'DSPs': 'dsp', 'URAM': 'uram'}
        for line in sec.group(1).splitlines():
            cells = [c.strip() for c in line.strip('|').split('|')]
            if cells and cells[0] in want:
                for i, s in enumerate(slrs):
                    out['per_slr'].setdefault(s, {})[want[cells[0]]] = int(cells[1 + i])
                    out['per_slr'][s][want[cells[0]] + '_pct'] = float(cells[1 + len(slrs) + i])
    for m in re.finditer(r'\|\s*(SLR\d) <-> (SLR\d)\s*\|\s*(\d+)\s*\|\s*\|\s*(\d+)\s*\|', txt):
        out['sll'][f'{m.group(2)}-{m.group(1)}'] = dict(used=int(m.group(3)), available=int(m.group(4)))
    return out

def parse_congestion(path):
    txt = open(path, errors='replace').read()
    per_node = defaultdict(lambda: dict(nets=0, cells=0))
    per_boundary = defaultdict(int)
    sec = txt.rsplit('SLR Net Crossing Reporting', 1)
    if len(sec) < 2: return {}
    hdr = re.search(r'\|\s*Cell Names\s*\|(.*?)\n', sec[1])
    cols = [c.strip() for c in hdr.group(1).strip().strip('|').split('|')] if hdr else []
    for line in sec[1].splitlines():
        if not line.startswith('|') or 'Cell Names' in line: continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        name, n = cells[0], int(cells[1])
        node = cell_to_node(name) or ('<top-control>' if name.startswith(('ap_', 'start_for', 'ctrl_')) else '<other>')
        per_node[node]['nets'] += n; per_node[node]['cells'] += 1
        for c, v in zip(cols[1:], cells[2:]):
            per_boundary[c] += int(v)
    windows = len(re.findall(r'^\|\s*(?:North|South|East|West)\s*\|', txt, re.M))
    return dict(crossing_nets_per_node=dict(sorted(per_node.items(), key=lambda kv: -kv[1]['nets'])),
                crossing_nets_per_boundary=dict(per_boundary), congestion_windows=windows)

def parse_crossing_csv(path):
    rows = []
    for line in open(path):
        parts = line.rstrip('\n').split(',')
        if len(parts) < 3 or parts[0] == 'slack': continue
        rows.append(dict(slack=float(parts[0]), source=parts[1], destination=parts[2],
                         src_node=cell_to_node(parts[1]), dst_node=cell_to_node(parts[2])))
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run_dir'); ap.add_argument('--out')
    a = ap.parse_args()
    fb = {'run_dir': a.run_dir}
    p = os.path.join(a.run_dir, 'timing_summary.rpt')
    if os.path.exists(p): fb['timing'] = parse_timing_summary(p)
    p = os.path.join(a.run_dir, 'utilization_slr.rpt')
    if os.path.exists(p): fb['utilization'] = parse_utilization_slr(p)
    p = os.path.join(a.run_dir, 'congestion.rpt')
    if os.path.exists(p): fb['crossing'] = parse_congestion(p)
    p = os.path.join(a.run_dir, 'crossing_paths.csv')
    paths = parse_crossing_csv(p) if os.path.exists(p) else fb.get('timing', {}).get('worst_paths', [])
    pairs = defaultdict(lambda: dict(count=0, worst_slack=0.0))
    for r in paths:
        if r['slack'] >= 0 or not r['src_node'] or not r['dst_node'] or r['src_node'] == r['dst_node']: continue
        k = f"{r['src_node']}->{r['dst_node']}"
        pairs[k]['count'] += 1; pairs[k]['worst_slack'] = min(pairs[k]['worst_slack'], r['slack'])
    fb['failing_node_pairs'] = dict(sorted(pairs.items(), key=lambda kv: kv[1]['worst_slack']))
    fb['internal_failing_nodes'] = sorted({r['src_node'] for r in paths
                                           if r['slack'] < 0 and r['src_node'] and r['src_node'] == r['dst_node']})
    js = json.dumps(fb, indent=1)
    if a.out: open(a.out, 'w').write(js)
    else: print(js)

if __name__ == '__main__':
    main()
