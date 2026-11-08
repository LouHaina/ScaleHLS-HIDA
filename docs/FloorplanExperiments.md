# Floorplan experiments (local branch fp-iter-local, on top of PR #4)

## 1. Question

Does an HLS-level floorplan in the style of PR #4 (ILP partitioning of dataflow
units across SLRs) give a better *routed* result than Vivado on its own, and
which part of the method causes the gain?

Two findings shape the design of the experiments:

* On ResNet18 t4_u2 (3% of xcvu37p) every SLR floorplan loses to no floorplan
  (WNS +1.9 ns none vs -3.1 ns PR #4 ILP): Vivado places the design in one SLR,
  and a floorplan can only add crossings. Floorplanning must be evaluated on
  designs that **have** to cross SLRs.
* The published gains (AutoBridge FPGA'21, TAPA) come mostly from **pipelining
  the SLR-crossing connections** once the floorplan is fixed, which Vivado may
  not do on its own (it cannot add latency). Pblocks alone mostly help routability.
  PR #4 describes crossing pipelining (design doc Sec. 6.3) but does not
  implement it. The experiments therefore separate *floorplan only* from
  *pipelining only* and from *both*.

## 2. Compute constraint and the strip setup

Full-device P&R of ResNet18 t16_u64 (804k LUT, 62% of xcvu37p) needs ~32 GB;
on the 16 GB laptop it paged continuously (~11k pages/s, <1 core used) and had
not finished routing after 9 h (stopped). Post-phys_opt WNS was -3.57 ns.

To keep each run within memory, a mid-size design (150-300k LUT) is confined
to a **vertical strip** of clock-region columns (e.g. X0-X2: ~180k LUT and
1170 DSP per SLR, ~540k LUT over the 3 SLRs). The design then has to span
2-3 SLRs, as a large design does on the full device, while the netlist and
P&R stay small. Every configuration gets the same strip; floorplans divide the
strip, the baseline gets the whole strip. Caveat: a strip has proportionally
fewer SLL columns per boundary, so crossings are scarcer than on the full
device (this makes crossings *more* critical, it does not favour the baseline).

## 3. Workloads

ResNet18 (`samples/pytorch/resnet18`), HIDA `-hida-pytorch-pipeline`, Vitis HLS
2023.2 at 3.33 ns, Vivado 2023.2 out-of-context synthesis, one shared
`post_synth.dcp` per workload.

| Id | Config | Strip | Target |
|---|---|---|---|
| W1 | the candidate (t4_u32 / t8_u32) whose real LUT is ~150-300k | X0-X2 | needs ~1.5-2 SLRs of the strip |
| W2 | same design, narrower strip (X0-X1), or the other candidate | X0-X1 | needs ~2.5 SLRs: high pressure |
| W3 | t16_u64, full device | full | only if a 64 GB+ machine is available; confirms the strip trend |

## 4. Configurations

| Id | Floorplan | Crossing pipelining | Purpose |
|---|---|---|---|
| B0 | none (strip only) | no | Vivado baseline |
| B0-SSI | none, SSI placer directives | no | strongest Vivado-only baseline |
| P4 | PR #4 `proto_floorplan.py` as is (C++ graph, proxy area, SLR pblocks, CONTAIN_ROUTING) | no | the PR's method |
| G1 | netlist-graph SLR ILP (bit-weighted edges from the synthesized netlist, measured area, top pinned to middle SLR, no CONTAIN_ROUTING) | no | improved floorplan |
| G2 | G1 + feedback from the worst paths of its own run (PR #4 doc Sec. 8) | no | iterative loop |
| R0 | none | yes, on all inter-unit connections | control: is the gain from registers alone? |
| P4+R | P4 | yes, on cut connections | PR method completed |
| G3 | G1/G2 | yes, on cut connections | full method |

The crossing-pipelining mechanism is chosen after the failing paths of B0 are
classified (step A2). Candidates, in order of safety: a register tree for the
shared reset; a register stage on quasi-static signals (buffer offsets and
other s_axilite arguments that are written before `ap_start`); register slices
on AXI channels; delayed ap_start/ap_done handshakes (needs a protocol argument
and RTL co-simulation). Any change that adds latency is checked by RTL
simulation against the unmodified design on a small configuration (t4_u2), and
its cycle-count cost is reported.

## 5. Controls against noise and bias

* Same checkpoint, clock (3.33 ns, 0.2 ns uncertainty), opt/place/phys_opt/route
  sequence, Vivado version and thread count (8) for every run.
* Vivado results move by a few hundred ps with any perturbation. Every
  configuration is run with several placer directives (Default,
  ExtraTimingOpt, SSI_SpreadLogic_high, SSI_BalanceSLLs). Report the best and the
  median over directives; a win must hold for the best-vs-best comparison, not a
  cherry-picked directive.
* Floorplans are produced without looking at the baseline's routed result,
  except G2, whose feedback comes from its own G1 run.

## 6. Procedure

A. Calibration (W1)
1. B0 with each directive, place stage (post-place checkpoint saved).
2. Classify B0's 100 worst paths: inter-SLR, top-controller/reset fan-out,
   intra-unit logic depth (logic vs route delay). This decides whether
   floorplanning or crossing pipelining can help at all.
3. Route B0 Default and B0 best; measure the post-place vs routed WNS gap.

B. Floorplan only: P4, G1 (then G2) with each directive, place stage.

C. Pipelining: R0, P4+R, G3, place stage.

D. Final: route the best directive of each configuration (and Default for each),
   on W1 and W2. Optional W3 on a large machine.

## 7. Metrics

WNS, TNS, failing endpoints, Fmax = 1000 / (3.33 - WNS) MHz, SLL usage per
boundary, per-SLR utilization, worst-path composition (inter-SLR share, logic
vs route delay, unit pairs), P&R runtime. For configurations that add latency:
cycles (HLS/co-sim) and execution time = cycles / Fmax.

## 8. Post-place estimates vs routed results

Post-place timing (placer + phys_opt) is used to screen variants (step B, C).
It models wire delay from real locations and usually tracks routed timing
closely, but it can be optimistic on congested designs (router detours). The
gap and whether the post-place ranking survives routing are measured in A3 and
D. Final claims use routed results only.

## 9. Success criterion

A floorplan-based configuration whose routed WNS (and, if latency changes,
execution time) beats the best B0 / B0-SSI result on W1 and W2, with the
ablation (R0, G1 vs G3) showing which part of the method produces the gain.
