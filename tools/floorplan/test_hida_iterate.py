#!/usr/bin/env python3
"""Dry run of hida_iterate.py: real compiler pass and feedback analyzer,
Vivado replaced by replaying recorded runs.

Replayed runs (ResNet18 t4_u16, strip X0-X2):
  original RTL        <- t4_u16/out_g1_Default_place   (control broadcast dominates)
  ctrl RTL            <- t4_u16_pc/out_g1_Default_place (FSM -> AXI gating dominates)
  ctrl+ungate_axi RTL <- same run with the worst paths rewritten into
                         intra-SLR unit-to-unit paths (placement_local)
Expected decisions: initial -> rtl:ctrl -> rtl:ungate_axi -> feedback+refine:band
-> feedback, then convergence (replayed WNS does not change).
"""
import csv, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hida_iterate as hi  # noqa: E402

REC = hi.ROOT
WORK = 'hida_mock'


def fake_netlist(base, work, passes):
    cfg = f'{work}/rtl_' + ('_'.join(passes) or 'orig')
    os.makedirs(os.path.join(REC, cfg), exist_ok=True)
    return cfg


def fake_place(cfg, fp, strip):
    out = os.path.join(REC, cfg, f'out_{fp}_Default_place')
    if os.path.exists(os.path.join(out, 'summary.txt')):
        return out
    rtl = cfg.split('rtl_')[1]
    src = {'orig': 't4_u16', 'ctrl': 't4_u16_pc'}.get(rtl, 't4_u16_pc')
    src = os.path.join(REC, src, 'out_none_Default_place' if fp == 'none' else 'out_g1_Default_place')
    os.makedirs(out, exist_ok=True)
    for f in ('summary.txt', 'paths_place.csv'):
        shutil.copy(os.path.join(src, f), out)
    if rtl == 'ctrl_ungate_axi':
        rows = list(csv.DictReader(open(os.path.join(out, 'paths_place.csv'))))
        for r in rows:
            r['start'] = 'grp_forward_node125_fu_1934/x_reg/C'
            r['end'] = 'grp_forward_node82_fu_2074/y_reg/D'
            r['inter_slr'] = '0'
        with open(os.path.join(out, 'paths_place.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    return out


hi.netlist, hi.place = fake_netlist, fake_place
shutil.rmtree(os.path.join(REC, WORK), ignore_errors=True)
sys.argv = ['hida_iterate.py', '--mlir', 'r18_t4_u16.mlir', '--base', 't4_u16', '--work', WORK, '--max-iters', '8']
hi.main()
rows = list(csv.DictReader(open(os.path.join(REC, WORK, 'iterations.csv'))))
actions = [r['action'] for r in rows]
print('actions:', actions)
expected = ['initial', 'rtl:ctrl', 'rtl:ungate_axi', 'feedback+refine:band', 'feedback']
ok = actions[:len(expected)] == expected and len(rows) <= len(expected) + 1
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
