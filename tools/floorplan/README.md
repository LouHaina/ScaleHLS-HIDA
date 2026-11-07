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
