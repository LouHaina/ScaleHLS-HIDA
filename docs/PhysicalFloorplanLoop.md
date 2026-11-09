# Compiler-driven iterative floorplanning

Builds on PR #4 (ILP SLR floorplanning of HIDA dataflow nodes, design doc
Sec. 8 "iterative loop"). The floorplan is decided by a compiler pass and is
refined with feedback from Vivado until it stops improving.

## Loop

```
 HIDA IR (r18_*.mlir)
   │
   ▼
 scalehls-opt -scalehls-physical-floorplan="device=.. strip=.. [feedback=fb_{k-1}.json]"
   │  nodes  : func.call of the top function, hls.axi.bundle (-> m_axi adapter),
   │           controller, extra netlist cells from the feedback
   │  sizes  : QoR estimate (iteration 0) / measured per cell (feedback)
   │  edges  : IR structure (iteration 0) / netlist bits (feedback) + critical pairs
   │  solve  : slots = SLRs or half-SLR bands, capacity rho, balance, stability
   │  decide : next action from the path classes of the previous run
   ▼
 it{k}.tcl (pblocks), it{k}.json (assignment, action, rtl_passes),
 IR annotated with hls.floorplan_slot
   │
   ▼
 netlist for rtl_passes (cached): HLS RTL -> pipeline_ctrl.py -> Vivado synthesis
   │
   ▼
 Vivado place (run_pnr.sh, place stage, strip)  +  same RTL without floorplan (baseline)
   │
   ▼
 make_feedback.py -> fb_k.json  ──────────────────────────────► next iteration
```

Driver: `tools/floorplan/hida_iterate.py --mlir r18_t4_u16.mlir --base t4_u16 --work <dir>`;
every iteration is appended to `<dir>/iterations.csv` with the baseline of the same RTL.

## Actions chosen by the pass

| Dominant class of the 100 worst paths | Action |
|---|---|
| `ctrl_broadcast`: unit -> all-subcalls-done -> ap_continue of every unit | RTL pass `ctrl` (register the condition) |
| `fsm_axi_gate`: top FSM state -> m_axi adapters | RTL pass `ungate_axi` (remove redundant single-user gating) |
| `placement_local`: long routes inside one SLR, slots are SLRs | refine slots to half-SLR bands, feed back critical pairs |
| `placement_xslr` | feed back critical pairs |
| `unit_internal`: logic inside one unit | stop (no floorplan can help) |

The loop stops on `stop` or when WNS improves by less than 0.02 ns twice.

## Solver

Same objective as `tools/floorplan/grid_floorplan.py` (CP-SAT):
`sum_e round(10 w_e) (|dx| + |dy| + slrPen |dslr|) + balance * 1000 * maxUtil + stability`,
load <= rho * capacity per slot and resource. The pass uses greedy construction,
move/swap/group-move local search (a node moves with the AXI adapters only it
uses) and iterated perturbation. On ResNet18 t4_u16 (150 cells, 530 edges,
3 SLR slots of a 3-column strip) it reaches the CP-SAT optimum
(2,284,748 vs 2,286,400, which rounds maxUtil up) in ~3 s.

## Current limits

* The RTL pipelining passes are applied to the Vitis HLS output
  (`pipeline_ctrl.py`); the pass chooses them, it does not emit them.
* The floorplan does not change the emitted C++; HLS runs once per design.
* Netlist names are mapped by Vitis naming (`grp_<func>_fu_<n>`, `axi_N_m_axi_U`).
