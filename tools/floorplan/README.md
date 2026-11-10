# Floorplan loop tools

Prototype pieces of the design in `docs/FloorplanAwareHIDA-Design.md`. They
run without a compiler build so the algorithm and the feedback path can be
tested on existing outputs.

| File | Role in the design | Status |
|---|---|---|
| `proto_floorplan.py` | Graph extraction (from HIDA-emitted C++) + ILP floorplanner (components B and C) | Tested on `samples/pytorch/resnet18/resnet18.cpp` |
| `parse_reports.py` | Vivado feedback parser (component E) | Tested on `data/run_*` |
| `run_vivado_iter.tcl` | One P&R iteration with machine-readable reports and optional incremental reuse | Needs Vivado to run |

Requires `pip install ortools`.

## Results on ResNet18 (tile 4, unroll 2), 3 SLRs

```
python3 tools/floorplan/proto_floorplan.py samples/pytorch/resnet18/resnet18.cpp \
    --top forward --slrs 3 --baseline-tcl data/run_0_2_1/floorplan_constraints.tcl
```

Graph: 24 units (top-level hierarchical nodes), 129 leaves, 105 on-chip buffer
edges, 30 token-stream edges. Top-level units are connected only by 1-bit token
streams; all wide edges are inside units.

| Floorplan | Cut buffer edges | Cut token edges | Bits per boundary | Units split across SLRs | Solve time |
|---|---|---|---|---|---|
| Existing BFS+FM (`data/run_0_2_1`) | 7 | 8 | 1305 / 1725 | 4 | - |
| ILP, unit level | 0 | 13 | 18 / 21 | 0 | 5.4 s (optimal) |
| ILP, leaf level | 0 | 11 | 18 / 15 | 0 | 7.7 s (optimal) |

Areas are proxies derived from the C++ (unrolled multiplies, local array
sizes); they are replaced by csynth numbers in the real flow.

## Feedback parser

```
python3 tools/floorplan/parse_reports.py data/run_0_2_1
```

gives WNS/TNS, per-SLR utilization, SLL usage per boundary, crossing nets per
HIDA node, failing node pairs (for example `forward_node156->forward_node97`
at -3.676 ns) and nodes with internal failures. Cell names are mapped to nodes
with the regex `(grp_)?<node>(_N)?(_fu_N|_U0)`.

## Running the loop with Vivado on Windows and the compiler on Linux

The loop is file based. Nothing in it needs the two tools on one machine.

1. Linux (compiler side) produces, per iteration `i`, a directory `iter_i/`
   with `design.cpp`, `floorplan.tcl`, `manifest.json`.
2. Windows (Vivado side) runs Vitis HLS and Vivado in batch mode on that
   directory and writes the reports into `iter_i/out/`:
   ```
   vitis_hls -f hls.tcl                    (csynth + export, once per design)
   vivado -mode batch -source run_vivado_iter.tcl ^
          -tclargs post_synth.dcp floorplan.tcl 4.0 iter_i\out [iter_{i-1}\out\post_route.dcp]
   ```
3. Linux parses `iter_i/out` with `parse_reports.py`, updates the model and
   writes `iter_{i+1}/`.

Three ways to share the directory, from simplest to most automated:

- WSL2 on the Windows machine: build the compiler inside WSL2 Ubuntu and keep
  the run directory under `/mnt/c/...`. A Python driver in WSL2 can call
  `/mnt/c/Xilinx/Vivado/2023.2/bin/vivado.bat -mode batch ...` directly, so
  the whole loop is one script. This is the recommended setup.
- Shared folder (SMB or OneDrive): the Linux driver writes `iter_i/`, then
  polls for `iter_i/out/ITERATION_DONE` (written by the Tcl at the end). A
  Windows batch file watches for new `iter_i/floorplan.tcl` and runs Vivado.
- SSH: enable OpenSSH Server on Windows and let the Linux driver run
  `ssh win "C:\\Xilinx\\Vivado\\2023.2\\bin\\vivado.bat -mode batch -source ..."`,
  copying files with `scp`. Fully automatic, no polling.

Vitis HLS only needs to run again when the C++ changes (granularity or stream
depth changes). Pure re-assignment iterations reuse the same post-synthesis
checkpoint and only re-run `run_vivado_iter.tcl`.

## Compiler-integrated flow (new passes)

Two passes implement components B and D of the design:

- `scalehls-export-dataflow-graph="output=graph.json top-func=forward"` runs
  after QoR estimation, recovers producer/consumer edges from calls that share
  a buffer or stream, marks the callees of the top function as units
  (`floorplan_unit`, `inline_off`, emitted as `#pragma HLS inline off`), tags
  every channel op with a stable `floorplan_id`, and writes `graph.json`.
- `scalehls-apply-floorplan="assignment=assignment.json"` sets `slr` on
  functions and `fifo_depth` on cut stream channels (emitted as
  `#pragma HLS stream depth=N`).

Both are reachable from the PyTorch pipeline:

```
scalehls-opt resnet18.mlir -hida-pytorch-pipeline="top-func=forward ... \
    floorplan-graph-output=graph.json floorplan-assignment=assignment.json"
```

Driver (`floorplan.py`):

```
floorplan.py solve graph.json --device devices/u280.json --out assignment.json \
    [--calibration calibration.json] [--feedback feedback.json --prev prev_assignment.json]
floorplan.py emit assignment.json --device devices/u280.json --out-tcl floorplan.tcl --out-manifest manifest.json [--soft]
floorplan.py calibrate graph.json --hls-dir <dir with forward_csynth.rpt> --out calibration.json
floorplan.py loop loop_config.example.json
```

`solve` collapses leaves into units, anchors AXI units to the memory SLR from
the device file, keeps on-chip buffer pairs grouped (`--allow-buffer-cut` to
relax), adds a throughput budget on dependency cycles, and, from the second
iteration on, locks units not implicated in failing paths and penalizes edges
whose unit pair failed timing. `emit` writes pblock (or `USER_SLR_ASSIGNMENT`)
constraints with regex cell matching per unit, a verification warning for
units that match no cells, and best-effort pinning of cut-stream FIFOs next to
their consumer. `calibrate` replaces estimator areas with the instance table
of the Vitis HLS csynth report (HIDA's estimator reports LUT = 0 and
over-estimates BRAM, so this step matters).

### Tested end to end (compiler build, ResNet18, tile 4, unroll 2, U280)

```
scalehls-opt resnet18.mlir -hida-pytorch-pipeline="top-func=forward loop-tile-size=4 \
    loop-unroll-factor=2 floorplan-graph-output=graph.json" > /dev/null
floorplan.py solve graph.json --device devices/u280.json --balance 0.3 --out assignment.json
scalehls-opt resnet18.mlir -hida-pytorch-pipeline="... floorplan-graph-output=graph.json \
    floorplan-assignment=assignment.json" | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives > design.cpp
floorplan.py emit assignment.json --device devices/u280.json --out-tcl floorplan.tcl --out-manifest manifest.json
```

- Exported graph: 24 units + memory pseudo-node, 172 functions, 290 edges
  (123 on-chip buffer edges inside units, 45 DRAM producer/consumer edges
  between units recovered through the AXI ports of the runtime `main`, 122
  AXI edges to the memory node).
- ILP: optimal in 0.3 s; 10/7/7 units per SLR; no on-chip buffer cut; 5 DRAM
  edges and 66 AXI edges cross a boundary (those carry no SLL wires).
- Applied IR: 172 functions carry `slr`; emitted C++ has `#pragma HLS inline off`
  on exactly the 24 units.
- lit: `test/Transforms/Floorplan` passes; the whole ScaleHLS suite is
  43 passed, 2 unsupported (Python bindings and Polygeist not built).

`calibrate`, `loop`, `hls_csynth.tcl` and `run_vivado_iter.tcl` need Vitis HLS
and Vivado and are untested here. The ILP balances on DSP until `calibrate`
supplies LUT/FF numbers, because HIDA's estimator reports LUT = 0.

### Objective: cut first, balance only on request

`solve` minimizes the weighted cut subject to the per-SLR capacity
(`--rho`, default 0.7). Balance is no longer part of the default objective:
`--beta` (default 0) is the cost of one percent of imbalance in units of
weighted wire bits. The earlier unnormalized term compared LUT counts with
wire bits, always won, and spread even a design that fits in a tenth of one
SLR over all of them, which can only add crossing delay. With the default, a
small design stays in one SLR and the result equals the baseline; the
floorplan only starts cutting when capacity forces it. `--balance X` still
caps every SLR at `(1 + X) / K` of the design and is meant for the forced
spread experiment, not for normal runs.

### Feedback

`solve --feedback fb0.json fb1.json ... --prev assignment.json` takes the
feedback of every earlier iteration, so penalties accumulate. A failing unit
pair that has no edge in the dataflow graph (the control handshake through the
top FSM: `ap_done`, `ap_sync_reg_*`, `loop_dataflow_output_count`) gets a
`ctrl` edge so the penalty has something to act on. A pair that failed in
`--group-after` iterations (default 2) is forced into one SLR. If the locks of
unaffected units keep a failing pair apart, the solve is repeated without
locks. Moving units only helps while co-location is possible; crossings that
must remain need registers (design document, section 6.3).

## Per-SLR resource budget (unit unroll factors)

Plain HIDA has one knob, `loop-unroll-factor`, and splits it over the units by
their share of the computation. With a budget, every unit gets its own factor,
chosen together with its SLR so that each SLR stays under a share of its
resources. The aim is a larger design that still fits and routes, measured in
frames per second, not a faster clock for the same design.

```
C='scalehls-opt samples/pytorch/resnet18/resnet18.mlir -hida-pytorch-pipeline="top-func=forward loop-tile-size=4 {extra} floorplan-graph-output={graph}" -o /dev/null'

# 1. compile at several factors (about 5 s each) and export a graph each time
floorplan.py sweep --compile "$C" --factors 1,2,4,8,16,32,64 --knob 2,8,32,64,128,256 --out-dir sweep

# 2. choose a factor and an SLR per unit; at most 60 percent of every SLR
floorplan.py budget sweep/all_*[0-9].json --baseline sweep/knob_*.json \
    --device devices/u280.json --rho 0.6 --out assignment.json --out-factors unit_factors.json

# 3. compile with the chosen factors, then check the result against the budget
scalehls-opt resnet18.mlir -hida-pytorch-pipeline="top-func=forward loop-tile-size=4 loop-unroll-factor=2 \
    floorplan-unit-factors=unit_factors.json floorplan-graph-output=final.json floorplan-assignment=assignment.json" \
  | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives > design.cpp
floorplan.py budget-check final.json assignment.json --device devices/u280.json --rho 0.6
floorplan.py emit assignment.json --device devices/u280.json --out-tcl floorplan.tcl --out-manifest manifest.json
```

How it works:

- `scalehls-parallelize-dataflow-node` accepts `unit-factors=<json>` (pipeline
  option `floorplan-unit-factors`): one factor per top-level dataflow node, in
  program order. A file written for a different number of units is rejected.
  The factor each unit was compiled with is recorded and exported in
  `graph.json` (`unit_index`, `unroll`).
- `budget` does not model area or latency. For every unit it reads both from
  the graphs in which that unit was compiled at that factor, so the candidates
  of a unit are the factors of the sweep. It minimizes the sum of the unit
  latencies (`--objective max`: the largest one), then area, then the cut. A
  factor that costs area without making the unit faster is not chosen.
- `--rho` is the share of every SLR that may be used; `--rho-slr 0.6,0.5,0.6`
  lowers one SLR, which is the feedback step after a run that was congested
  there. `--assignment` keeps an earlier SLR assignment and only re-chooses
  factors. The solver keeps `--margin` (5 percent) free, because units that
  share an external buffer influence each other a little and HIDA's output
  varies slightly from run to run.
- `--baseline` graphs are compiles with the plain knob. For each one `budget`
  reports whether it fits the same budget and its latency.

### Result on ResNet18 (tile 4, U280 device file, 24 units)

All numbers are HIDA's own estimates (QoR estimation pass), not HLS or Vivado
results. Latency is the sum of the unit latencies in cycles.

| Share per SLR | Best plain knob that fits | Its latency | Unit factors | Recompiled latency | Ratio |
|---|---|---|---|---|---|
| 0.6 | `loop-unroll-factor=64` | 3.09e9 | 16 64 64 64 8 64 64 64 64 1 64 64 4 4 64 4 4 1 1 1 1 1 1 64 | 2.26e9 | 1.37x |
| 0.3 | `loop-unroll-factor=32` | 3.21e9 | 16 8 64 8 8 8 8 64 8 4 8 8 4 4 64 4 4 1 1 64 1 1 64 1 | 2.36e9 | 1.36x |
| 0.1 | `loop-unroll-factor=2` | 3.66e9 | 9 64 64 4 2 4 2 64 4 1 4 2 64 64 64 4 64 1 1 64 1 64 64 64 | 2.62e9 | 1.39x |

In all three cases the recompiled design passed `budget-check`, and its
latency was within 4 percent of what `budget` expected. A few units differed
from the expected DSP count by more than 20 percent (two at 0.6, three at 0.3),
which is the neighbour effect the margin is for.

What this does and does not show:

- It shows that the estimator's own numbers favour an uneven distribution: the
  plain knob gives most of the factor to the first convolution, and several
  later units get slower, by the estimate, when unrolled further.
- It does not show a hardware gain. The estimator reports LUT = 0 and
  over-estimates BRAM, so here the budget is bounded by estimated BRAM and DSP
  only. The table has to be repeated with Vitis HLS areas and then with routed
  designs (`eval_sweep.py`) before it is a result.
- With `loop-tile-size=4` the units stop growing beyond a factor of about 16,
  which is why the design stays small. Larger designs need a larger tile size.

Tests: `python3 tools/floorplan/test_budget.py`, and the lit tests
`test/Transforms/Dataflow/parallelize-unit-factors.mlir` and
`test/Transforms/Floorplan/export-unit-factors.mlir`.

## Evaluation against the unconstrained baseline

`eval_sweep.py` runs every arm through the same Tcl, clock period and placer
directives and compares medians:

```
eval_sweep.py run       eval_config.example.json     (skips finished runs)
eval_sweep.py summarize eval_config.example.json     (table + <out>/summary.csv)
eval_sweep.py summarize --dirs data/run_0_2 data/run_0_2_1 --period 8.0
```

Arms: `baseline` (plain checkpoint, no constraints), `pipeline_only`
(floorplan checkpoint, no constraints), `floorplan_only` (plain checkpoint,
constraints), `full`. `run_vivado_iter.tcl` takes `none` as the constraint
file for the unconstrained arms and a place directive as sixth argument, and
writes `status.txt` so that placement and routing failures are counted instead
of lost. Fmax is `1000 / (period - WNS)` post-route; failed runs are listed
separately and never averaged in as 0 MHz. Use one config per design size and
sweep the size (tile and unroll factors) from about 10 to 70 percent of the
device: published gains (AutoBridge, PASTA) appear only for large designs and
are about zero for small ones. `summarize` is tested on the reports under
`data/`; `run` needs Vivado and is untested here.
