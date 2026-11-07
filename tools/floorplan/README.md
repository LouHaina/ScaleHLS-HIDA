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
