#!/usr/bin/env python3
"""Register the "all sub-calls done" condition of the HLS top-level FSM.

In the HIDA designs Vitis HLS starts all top-level units in one FSM state and
leaves it when every unit is done:

    ap_block_stateN_on_subcall_done = ~(done_0 & done_1 & ... & done_k)

where done_i is a unit's (combinational) ap_done or its sync register.  The same
signal drives ap_continue of every unit, the clear of every ap_sync_reg and
the FSM, so the critical path runs from logic inside one unit, through an AND
over all units, back into every unit (on ResNet18 t4_u16 in a 3-column strip:
the 100 worst paths, ~4.7 ns of routing, fanout ~374k).

The rewrite computes the condition into a register, one cycle later:

    all_done_q <= stateN & ~block_comb & ~all_done_q
    block       = ~all_done_q

Why it is safe (cost: one cycle per top-level invocation of stateN):
  * every term of the AND stays true until the units receive ap_continue:
    ap_ctrl_chain units hold ap_done until ap_continue, the ap_sync_reg_* flags
    hold until cleared by ~block, and ap_ctrl_hs units without a sync register
    report ap_done while idle and are only restarted from another FSM state;
  * all_done_q is only set from a cycle spent in stateN, so it is never stale
    from a previous invocation, and it clears itself in the cycle it is used,
    so ap_continue is a single-cycle pulse as before.
The ap_ready / ap_start handshake is left untouched.

Second rewrite (--ungate-axi): Vitis gates every m_axi handshake signal of a
unit with the FSM state(s) in which that unit runs,

    axi_N_ARVALID = (state1 | state2) ? unit_ARVALID : 0

When the bundle has a single user (the case for all bundles of the HIDA
designs), the gate is redundant: the unit is idle outside those states and an
idle HLS unit holds its VALIDs low, and a READY asserted with no outstanding
transaction transfers nothing.  The gate makes the top FSM state register a
driver of every AXI adapter (critical after the first rewrite: ~95% of the
worst paths, SLR crossings), so it is removed.  Bundles with more than one
user are left alone.

usage: pipeline_ctrl.py <in forward.v> <out forward.v> [--ungate-axi]
"""
import re, sys

src, dst = sys.argv[1:3]
v = open(src).read()
states = sorted(set(re.findall(r'ap_block_(state\d+)_on_subcall_done', v)))
if not states:
    sys.exit('no ap_block_stateN_on_subcall_done in ' + src)
add = []
for st in states:
    blk = f'ap_block_{st}_on_subcall_done'
    comb = f'{blk}_comb'
    q = f'{st}_all_done_q'
    n = len(re.findall(rf'^\s*{blk}\s*=', v, flags=re.M))
    if n != 1:
        sys.exit(f'{blk}: expected one combinational assignment, found {n}')
    v = re.sub(rf'^(\s*){blk}(\s*=)', rf'\1{comb}\2', v, count=1, flags=re.M)
    decl = f'reg    {blk};'
    if v.count(decl) != 1:
        sys.exit(f'{blk}: declaration not found')
    v = v.replace(decl, f'{decl}\nreg    {comb};\nreg    {q};')
    add += ['always @ (posedge ap_clk) begin',
            f'    if (ap_rst_n_inv == 1\'b1) {q} <= 1\'b0;',
            f'    else {q} <= (1\'b1 == ap_CS_fsm_{st}) & (1\'b0 == {comb}) & ~{q};',
            'end',
            f'always @ (*) {blk} = ~{q};']
    print(f'{st}: registered {blk}')
if '--ungate-axi' in sys.argv:
    pat = re.compile(r"always @ \(\*\) begin\n\s*if \(\(\(1'b1 == ap_CS_fsm_state\d+\)( \| \(1'b1 == ap_CS_fsm_state\d+\))*\)\) begin\n"
                     r"\s*(axi_\d+_\w+) = (\S+);\n\s*end else begin\n\s*\2 = 1'b0;\n\s*end\nend")
    users = {}
    for m in pat.finditer(v):
        users.setdefault(m.group(2), []).append(m)
    n = 0
    for x, ms in users.items():
        if len(ms) != 1 or len(re.findall(rf'^\s*{x} = ', v, flags=re.M)) != 2:
            continue  # more than one user or other assignments: keep the gate
        m = ms[0]
        v = v.replace(m.group(0), f'always @ (*) begin\n    {x} = {m.group(3)};  // pipeline_ctrl.py: ungated\nend', 1)
        n += 1
    print(f'ungated {n} single-user m_axi handshake signals')
i = v.rindex('endmodule')
v = v[:i] + '// pipeline_ctrl.py: registered sub-call completion\n' + '\n'.join(add) + '\n\n' + v[i:]
open(dst, 'w').write(v)
