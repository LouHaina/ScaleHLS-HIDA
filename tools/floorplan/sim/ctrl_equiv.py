#!/usr/bin/env python3
"""Control-protocol simulation of the HLS top level with stubbed sub-modules.

Every module instantiated by the top (dataflow units, m_axi adapters, s_axilite
blocks, buffers) is replaced by a stub with the same ports:
  * units implement the ap_ctrl handshake of the real modules
      - ap_ctrl_chain (has ap_continue): accept ap_start when idle (ap_ready),
        run a deterministic pseudo-random latency, then hold ap_done until
        ap_continue;
      - ap_ctrl_hs: ap_done/ap_ready pulse when finished, ap_done also while idle
        and ap_start is low (as the Vitis-generated sequential modules do);
    and log every start / done / continue, plus protocol violations (start
    while a done is pending, continue without done);
  * the s_axilite block drives ap_start from the testbench and logs top done;
  * everything else drives zeros.
The top runs N invocations. The logs of the original and of the rewritten top
(pipeline_ctrl.py) are compared: same per-unit event sequences, same number of
top completions, no violations; the cycle count may differ by the inserted
register stages.

usage: ctrl_equiv.py <verilog dir> <work dir> <forward.v>... (first = reference)
Runs Vivado xvlog/xelab/xsim through cmd.exe (work dir must be under /mnt/d).
"""
import os, re, subprocess, sys
from collections import defaultdict

vdir, work, tops = sys.argv[1], sys.argv[2], sys.argv[3:]
NINV = 3
os.makedirs(work, exist_ok=True)


def win(p):
    p = os.path.abspath(p)
    assert p.startswith('/mnt/d/'), p
    return 'D:\\' + p[len('/mnt/d/'):].replace('/', '\\')


def module_header(path, mod):
    """Port list, parameters and port declarations of <mod> in <path>."""
    txt = open(path, errors='replace').read()
    m = re.search(rf'^module\s+{mod}\b(.*?)^endmodule', txt, re.M | re.S)
    body = re.sub(r'//[^\n]*', '', m.group(1))
    head_end = body.index(');') + 2
    ports_txt = body[:head_end]
    decls = []
    for st in body[head_end:].split(';'):
        s = ' '.join(st.split())
        if re.match(r'(parameter|localparam)\b', s):
            decls.append(s + ';')
        elif re.match(r'(input|output|inout)\b', s):
            decls.append(re.sub(r'\b(reg|wire)\b', '', s, count=1) + ';')
    outs = []
    # ANSI-style headers declare the ports inside the port list
    for mo in re.finditer(r'output\s+(?:wire|reg)?\s*(\[[^\]]+\])?\s*(\w+)', ports_txt):
        outs.append(mo.group(2))
    for d in decls:
        mo = re.match(r'output\s*(\[[^\]]+\])?\s*(.+);', d)
        if mo:
            outs += [o.strip() for o in mo.group(2).split(',')]
    return ports_txt, decls, outs


ref = open(tops[0]).read()
insts = re.findall(r'^(forward_\w+)\s+(?:#\((?:.|\n)*?\)\s*)?(\w+)\s*\(', ref, re.M)
types = sorted(set(t for t, _ in insts))
stub_lines = ['`timescale 1 ns / 1 ps']
for uid, t in enumerate(types):
    f = os.path.join(vdir, t + '.v')
    ports, decls, outs = module_header(f, t)
    is_unit = t.startswith('forward_forward_node')
    is_ctrl = 'ctrl_s_axi' in t
    lines = [f'module {t}{ports}'] + [d if d.endswith(';') else d + ';' for d in decls]
    special = set()
    if is_unit:
        chain = 'ap_continue' in ports
        special = {'ap_done', 'ap_ready', 'ap_idle'}
        lines += [
            'reg busy = 0, pend = 0, fin = 0; reg [15:0] cnt = 0; integer inv = 0;',
            f'localparam UID = {uid};',
        ]
        if chain:
            lines += [
                'assign ap_ready = ap_start & ~busy & ~pend;',
                'assign ap_done = pend;',
                'assign ap_idle = ~busy & ~pend;',
                'always @(posedge ap_clk) begin',
                '  if (ap_rst) begin busy <= 0; pend <= 0; end else begin',
                '    if (ap_start & pend) $display("EV %m viol_start_while_done");',
                '    if (ap_ready) begin busy <= 1; cnt <= ((UID*37 + inv*11) % 50) + 3; inv = inv + 1;'
                ' $display("EV %m start"); end',
                '    else if (busy) begin if (cnt == 0) begin busy <= 0; pend <= 1; $display("EV %m done"); end'
                ' else cnt <= cnt - 1; end',
                '    if (ap_continue & pend) begin pend <= 0; $display("EV %m cont"); end',
                '  end',
                'end',
            ]
        else:
            lines += [
                'assign ap_ready = fin;',
                'assign ap_done = fin | (~busy & ~ap_start);',
                'assign ap_idle = ~busy;',
                'always @(posedge ap_clk) begin',
                '  if (ap_rst) begin busy <= 0; fin <= 0; end else begin',
                '    fin <= 0;',
                '    if (~busy & ap_start & ~fin) begin busy <= 1; cnt <= ((UID*37 + inv*11) % 50) + 3;'
                ' inv = inv + 1; $display("EV %m start"); end',
                '    else if (busy) begin if (cnt == 0) begin busy <= 0; fin <= 1; $display("EV %m done"); end'
                ' else cnt <= cnt - 1; end',
                '  end',
                'end',
            ]
    elif is_ctrl:
        special = {'ap_start'}
        lines += [
            'reg start_r = 0;',
            'assign ap_start = start_r;',
            'always @(posedge ACLK) begin',
            '  if (ap_ready) start_r <= 0;',
            '  if (ap_done) $display("EV top done");',
            'end',
        ]
    for o in outs:
        if o not in special:
            lines.append(f'assign {o} = 0;')
    lines.append('endmodule')
    stub_lines += lines
open(os.path.join(work, 'stubs.v'), 'w').write('\n'.join(stub_lines) + '\n')

ctrl_inst = next(i for t, i in insts if 'ctrl_s_axi' in t)
tb = f'''`timescale 1 ns / 1 ps
module tb;
reg clk = 0, rstn = 0; always #1 clk = ~clk;
forward dut(.ap_clk(clk), .ap_rst_n(rstn));
integer k, cyc = 0, t0;
always @(posedge clk) cyc = cyc + 1;
initial begin
  repeat (10) @(posedge clk); rstn = 1; repeat (5) @(posedge clk);
  for (k = 0; k < {NINV}; k = k + 1) begin
    t0 = cyc;
    @(negedge clk) dut.{ctrl_inst}.start_r = 1;
    @(posedge clk); while (!(dut.ap_done === 1'b1)) @(posedge clk);
    $display("EV top invocation %0d cycles %0d", k, cyc - t0);
    repeat (3) @(posedge clk);
  end
  $display("EV end");
  $finish;
end
initial begin #2000000 $display("EV timeout"); $finish; end
endmodule
'''
open(os.path.join(work, 'tb.v'), 'w').write(tb)

results = {}
for i, top in enumerate(tops):
    d = os.path.join(work, f'v{i}')
    os.makedirs(d, exist_ok=True)
    cmd = (f'cmd.exe /c "cd /d {win(d)} && D:\\Vivado\\2023.2\\bin\\xvlog.bat --nolog '
           f'{win(os.path.join(work, "stubs.v"))} {win(top)} {win(os.path.join(work, "tb.v"))} && '
           f'D:\\Vivado\\2023.2\\bin\\xelab.bat --nolog -debug off tb -s tb_snap && '
           f'D:\\Vivado\\2023.2\\bin\\xsim.bat --nolog tb_snap -R"')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    log = r.stdout + r.stderr
    open(os.path.join(d, 'sim.log'), 'w').write(log)
    ev = [l.split('EV ', 1)[1].strip() for l in log.splitlines() if 'EV ' in l]
    if not ev:
        print(f'{top}: no events, see {d}/sim.log')
        print('\n'.join(l for l in log.splitlines() if 'ERROR' in l)[:2000])
        sys.exit(1)
    results[top] = ev

def per_unit(ev):
    u = defaultdict(list)
    for e in ev:
        parts = e.split()
        if parts[0].startswith('tb.dut.'):
            u[parts[0]].append(parts[1])
    return u

ref_ev = results[tops[0]]
ref_u = per_unit(ref_ev)
ok = True
for top, ev in results.items():
    u = per_unit(ev)
    viol = [e for e in ev if 'viol' in e or 'timeout' in e]
    inv = [e for e in ev if e.startswith('top invocation')]
    done = sum(1 for e in ev if e == 'top done')
    same = u == ref_u
    seq_ok = all(s == ['start', 'done', 'cont'] * NINV or s == ['start', 'done'] * NINV for s in u.values())
    print(f'{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(top)))))}: '
          f'{len(u)} units, invocations {[e.split()[-1] for e in inv]} cycles, top_done {done}, '
          f'violations {len(viol)}, per-unit sequences ok {seq_ok}, identical to reference {same}')
    ok &= same and seq_ok and not viol and len(inv) == NINV
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
