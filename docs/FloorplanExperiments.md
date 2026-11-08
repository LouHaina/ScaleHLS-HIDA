# Floorplan experiments (local branch fp-iter-local, on top of PR #4)

## Why a new workload

On ResNet18 at tile 4 / unroll 2 (41.7k LUT, 502 DSP, 3% of xcvu37p), Vivado with
no floorplan places the whole design in one SLR and reaches WNS +1.9 ns at 8 ns.
Every SLR floorplan (old BFS+FM pass: -3.7 ns, PR #4 ILP: -3.1 ns) can only add
crossings. Floorplanning has to be evaluated on designs that cannot fit in one SLR.

## Workloads

ResNet18 from `samples/pytorch/resnet18/resnet18.mlir`, compiled with
`-hida-pytorch-pipeline="top-func=forward loop-tile-size=T loop-unroll-factor=U"`.

| Config | HIDA QoR DSP estimate | Expected real DSP (2x, calibrated on t4_u2: 245 est vs 502 real) | xcvu37p |
|---|---|---|---|
| t8_u64 | 1886 | ~3.8k | 42% DSP, > 1 SLR (3008 DSP each) |
| t16_u64 | 2664 | ~5.3k | 59% DSP, needs 2+ SLRs |

Real numbers come from Vitis HLS csynth and Vivado synthesis.

Flow: scalehls-opt -> scalehls-translate (`-emit-vitis-directives`) -> Vitis HLS 2023.2
csynth at 3.33 ns (300 MHz) -> Vivado 2023.2 out-of-context synthesis -> one
post-synthesis checkpoint per workload, shared by every configuration below.

## Configurations

| Id | Floorplan | Notes |
|---|---|---|
| B0 | none | Vivado default placement |
| P4 | PR #4 `proto_floorplan.py`, unit level, SLR pblocks | graph from the emitted C++, proxy areas |
| G1 | netlist-graph SLR ILP | graph from post-synthesis netlist, measured per-unit area, SLR capacity cap, units never split, top controller pinned to the middle SLR, no CONTAIN_ROUTING |
| G2 | G1 + Vivado feedback | failing paths of the G1 run mapped to unit pairs, re-solved (PR #4 doc Sec. 8) |

Each configuration runs with placer directives Default and ExtraTimingOpt
(SSI_SpreadLogic_high where time allows). Same checkpoint, same clock
(3.33 ns, 0.2 ns uncertainty), same opt/place/phys_opt/route sequence.

## Metrics

WNS and TNS at 3.33 ns, estimated Fmax = 1000 / (3.33 - WNS) MHz, SLL usage per
SLR boundary, per-SLR utilization, number of failing endpoints, P&R runtime.

## Success criterion

A floorplan whose WNS beats B0 under the same directive(s).
