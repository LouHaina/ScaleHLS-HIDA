# Floorplan-Aware HIDA: Design for SLR-Level Place-and-Route Optimization

Status: design proposal (thesis). Target: ScaleHLS-HIDA fork, Vitis HLS + Vivado 2023.2, multi-SLR Alveo devices.

## 1. Summary

Goal: make the HIDA compiler aware of the die structure (super logic regions, SLRs) of modern FPGAs so that every dataflow node is assigned to an SLR, per-SLR resources are balanced, the inter-SLR edges are cheap and pipelined, and Vivado place-and-route (P&R) reaches a higher clock frequency than vanilla HIDA at the same throughput. The method must be iterative: Vivado results (per-SLR utilization, SLL crossing counts, failing paths) flow back into the compiler model and re-partitioning.

The project is practical, but only under three conditions that the current prototype does not yet meet:

1. The partition graph must be the real dataflow graph (producer-consumer edges through buffers and streams), not the call graph. Today's pass sees a star rooted at `forward` and its cut cost is almost meaningless.
2. Every cut edge must be made latency-insensitive and registered. The worst path in the existing runs (data/run_0_2_1) is a 11.4 ns control net between two hierarchical nodes (`grp_forward_node156 -> grp_forward_node97`, `loop_dataflow_output_count`), 96% routing delay. Vitis HLS does not pipeline those wires for you.
3. The benchmark must be large enough to need more than one SLR. The existing runs use 2 to 9 percent of each SLR's CLBs. For such a design, any floorplan can only add crossing delay, so "better than HIDA" is unreachable by construction. The evaluation must use unroll and tile factors that push the design to roughly 40 to 70 percent of the device.

The novelty relative to AutoBridge/TAPA/RapidStream (post-HLS, RTL-level) and StreamTensor (ILP die assignment inside a different compiler) is: SLR assignment is decided at the HIDA IR level, jointly with dataflow granularity, buffer type and stream depth; names are compiler-controlled so the RTL hierarchy is matched exactly; and a closed loop feeds Vivado measurements back into the node area model, edge penalties and SLR capacities. No published work closes that loop at the HLS-compiler level (see Section 9).

## 2. What exists today and what is wrong with it

Code: `lib/Transforms/LayoutPartitionEnhanced.cpp` (881 lines), wired at the end of `hida-pytorch-pipeline` after `QoREstimation` (`lib/Transforms/Passes.cpp`).

| Aspect | Current behaviour | Problem |
|---|---|---|
| Graph | `func::FuncOp` call graph; edge weight = sum of CallOp operand bitwidths | After `ConvertDataflowToFunc`, every node is called from `forward`. Sibling nodes are connected by shared memrefs/streams, not by calls, so almost all real edges are missing and FM gains are noise. |
| Node areas | `ResourceAttr` from the estimator; FF := LUT; BRAM forced to 0 | BRAM is over-estimated (272 vs ~0), FF not modeled. Balancing on wrong numbers. |
| Algorithm | BFS from top, then FM moves to adjacent SLRs | Fine as fallback, but no SLL budget, no per-boundary cost, no anchors for memory ports. |
| Device | Capacity table for xcvu37p (3 SLRs), README says U250 with 4 SLRs | Inconsistent; capacity numbers are flat totals, not usable area after the shell. |
| Cut edges | Nothing is done to them | Cross-SLR handshake and data wires are unregistered. This is the actual cause of the negative slack in the runs. |
| Naming | `<prefix>_node<N>` from a counter; `#pragma HLS inline` on single-loop nodes; Tcl matches `*name*_U0` then `*name*` | Counter order changes with any transform change; inlined nodes vanish; Vitis names nested loops by C++ line number (`dataflow_in_loop_VITIS_LOOP_4352_1`). Glob fallback over-matches (`*node1*` matches node10..node199). |
| Feedback | `run.tcl` writes `utilization_slr.rpt`, `util_pblock_*.rpt`, `timing_summary.rpt`, `congestion.rpt` | Reports are produced but never parsed or fed back. |

Measured facts from `data/` (ResNet18, tile 4, unroll 2, xcvu37p, 8 ns clock):

- WNS across runs with near-identical constraints: +1.33, +0.65, +0.41, +0.11, -0.87, -1.73, -3.68, -5.95, -7.28 ns. Vivado variance dominates; single-run comparisons are not evidence.
- SLR crossing nets: 284 to 800, of which only 2 to 3 percent of SLL capacity. Crossings are not capacity-limited; they are latency-limited.
- Per-SLR CLB use: 3.7 / 5.1 / 8.9 percent. The design fits in a fraction of one SLR.

## 3. Design overview

```
         HIDA front-end (unchanged)
                  |
   dataflow IR: schedule / node / buffer / stream  (levels, taps, depths)
                  |
   [A] StableNaming + PartitionUnit selection      (new, before ConvertDataflowToFunc)
                  |
   ConvertDataflowToFunc, directives, QoREstimation (existing)
                  |
   [B] DataflowGraphExport  ---> graph.json         (new analysis: nodes, edges, areas, anchors)
                  |                      |
                  |          [C] Floorplanner (Python, OR-Tools CP-SAT / CBC ILP)
                  |                      |   assignment.json  (node -> SLR, edge -> pipeline depth)
   [D] ApplyFloorplan  <-----------------+
       - annotates funcs with slr / crossing attrs
       - rewrites cut edges: buffer->stream, depth, register
       - emits floorplan.tcl + manifest.json (cell regexes)
                  |
   EmitHLSCpp (+ inline off, stable names)  -> Vitis HLS csynth -> Vivado P&R
                  |
   [E] FeedbackParser: csynth.rpt, report_utilization -slr, report_slr_crossing,
       get_timing_paths -inter_slr, report_design_analysis
                  |
   [F] ModelUpdate: per-node area calibration, edge timing penalties,
       SLR capacity caps, node locks  ->  back to [C] (or [A] for granularity changes)
```

Components A to D live in the compiler (C++ passes plus one Python driver). E and F are a Python package (`tools/floorplan/`) that also orchestrates Vitis HLS and Vivado. Keeping the solver and the loop in Python is deliberate: the ILP is 25 to 200 nodes, solves in seconds, and iteration logic changes often during a thesis.

## 4. Partition units and stable naming (component A)

### 4.1 Granularity

HIDA's structure gives natural cut points. In ResNet18 the top `forward` is a dataflow region of 24 hierarchical nodes (one per layer group). Each of those contains 5 to 8 leaf nodes connected by on-chip ping-pong buffers; the 24 top-level nodes communicate only through DRAM buffers (m_axi) and 1-bit token streams. So:

- Default partition unit = top-level hierarchical node (level-1 node under the top schedule). Its leaf nodes travel with it. This keeps BRAM-channel producer/consumer pairs together (the same rule PASTA uses) and keeps the 13 inlined leaf nodes irrelevant to matching.
- A unit that exceeds the per-SLR cap (after calibration) is split along its internal buffer edges: its leaf nodes become units and the buffer edge that is cut is converted to a stream (Section 6). This is the IR-level answer to TAPA's "two kernels in one slot" failure.
- A unit may also be merged with its single predecessor if both are small and the edge between them is wide.

### 4.2 Stable, compiler-owned names

Replace the counter in `ConvertNodeToFunc` (`ConvertDataflowToFunc.cpp:188`) with a name derived from the node's structural position: `<top>_l<level>_s<schedule-index>_n<ordinal-in-topological-order>` plus a short hash of the node's loop nest signature (trip counts, operand types). The name is stored as a `hls.uid` string attribute on the `NodeOp` before conversion and carried onto the `FuncOp`. Two compiles with the same frontend input and the same options produce the same uid; unrelated option changes (unroll factor of another node) do not shift the numbering.

Rules for emission:

- Every partition unit gets `#pragma HLS inline off` (new attribute `inline_off`, emitted in `emitFunctionDirectives`). The existing `inline` attribute stays for leaf nodes that are not units.
- Names are kept short (under 60 characters) so Vitis does not truncate them; set `config_compile -name_max_length 80` in the HLS script.
- The manifest records, for each unit, the regexes that identify its RTL instance:
  - dataflow process instance: `.*/(<uid>)(_[0-9]+)?_U0$`
  - plain call instance: `.*/grp_(<uid>)(_[0-9]+)?_fu_[0-9]+$`
  - and the module names `<top>_<uid>`, `<top>_<uid>_s`.
- A `verify_floorplan.tcl` step runs after `synth_design`: for each unit, count matched cells and LUTs; abort with a clear message if a unit matches zero cells or if two units match the same cell. The manifest's expected LUT count (from csynth) is compared to the matched LUT count; a mismatch over 30 percent flags a hierarchy problem (flattening or inlining).
- `synth_design -flatten_hierarchy rebuilt` is kept (default); additionally set `KEEP_HIERARCHY = SOFT` on all unit instances in the Tcl so names survive even with aggressive synthesis options.

This removes the name-matching problem for everything we constrain. Sub-blocks named by Vitis line numbers live under the unit's instance and are covered by the unit's prefix.

## 5. Graph export (component B)

A new analysis pass `scalehls-export-dataflow-graph` runs after `QoREstimation` and writes `graph.json`:

- Nodes: uid, unit flag, parent unit, estimated LUT/FF/DSP/BRAM/URAM (estimator; replaced by calibrated values after the first HLS run), latency and II from `TimingAttr`, dataflow level.
- Edges: built from shared SSA values between `CallOp`s in the same dataflow function (two calls that both take the same memref or stream operand are connected; the producer is the one with the output position). Each edge carries: kind (on-chip buffer, stream, token stream, DRAM/axi), element bitwidth, depth, total transfer volume (elements per invocation, from the memref shape and `TimingAttr`), and level distance.
- Anchors: m_axi bundles and the s_axilite control port, each with the SLR of the memory controller they bind to (HBM on U280 is at the bottom of SLR0; DDR on U250 has one controller per SLR). These vertices are pre-assigned.
- Cycles: SCCs of the directed unit graph (feedback through DRAM or token streams); each cycle is listed with its token count so the floorplanner can bound how many registers it may add on that cycle (Section 6.3).

Edge weight used by the solver: `w_e = bits_e * k_kind`, with `k_stream = 1`, `k_buffer = 4` (a cut buffer becomes a stream plus a copy, so it is expensive), `k_token = 0.1`, `k_axi = 0` (AXI already goes through the interconnect; its cost is handled by the anchor).

## 6. Floorplanner (component C)

### 6.1 Model

- K SLRs in a line, 0..K-1. Boundary b is between SLR b and b+1.
- Per-SLR usable capacity `C_k^r` per resource r in {LUT, FF, DSP, BRAM, URAM}, computed from the device's clock-region table minus the shell/static region and minus the Laguna columns' reduced CLB counts (adopt AutoBridge's `CR_AREA` approach; the current flat numbers in `getPartitionCapacity()` are replaced by a device JSON). Per-boundary SLL capacity `S_b` (23040 on xcvu37p per `report_utilization -slr`).
- Utilization ratio `rho` (start 0.6, raise in steps of 0.05 until feasible, cap 0.8).

### 6.2 ILP

Variables: `y_{v,k}` binary, one-hot per node; integer `p_v = sum_k k*y_{v,k}`; `d_e >= p_src - p_dst`, `d_e >= p_dst - p_src` (number of boundaries crossed by edge e); `c_{e,b}` binary = edge e crosses boundary b (linearized as in AutoBridge's `ILPUtilities.py`).

Objective: minimize `sum_e w_e * d_e  +  alpha * sum_e pen_e * d_e  +  beta * imbalance`, where `pen_e` is the timing penalty learned from feedback (Section 8) and imbalance is `max_r (max_k U_k^r - min_k U_k^r)` over normalized utilizations (linearized with auxiliary variables).

Constraints:

- Capacity: `sum_v y_{v,k} * a_v^r <= rho * C_k^r` for all k, r.
- SLL budget: `sum_e bits_e * c_{e,b} <= gamma * S_b` for all b (gamma 0.5).
- Anchors: `y_{v,k} = 1` for memory-port vertices.
- Grouping: `p_u = p_v` for unit pairs connected by on-chip buffer edges that the user or the splitter marked as unsplittable.
- Adjacency: `d_e <= 1` for stream edges by default (crossing two boundaries doubles pipeline depth and routes through a middle SLR); relaxed to `d_e <= 2` only if infeasible.
- Cycle budget: for each cycle C with `tok_C` tokens in flight (sum of FIFO/token depths), `sum_{e in C} 2*d_e <= tok_C - 1` so the added register latency never lowers the cycle's throughput below 1 (standard latency-insensitive bound `throughput <= tokens/length`).
- Locks (iteration >= 1): `y_{v,k_prev} = 1` for nodes not implicated in failing paths, to maximize incremental P&R reuse.

Solver: OR-Tools CP-SAT (free, good on small ILPs) with a 60 s limit; fallback to the existing BFS+FM heuristic if no feasible solution. For K <= 4 and up to 200 units this solves in seconds (AutoBridge reports the same scale).

### 6.3 Cut-edge treatment (the part the current prototype lacks)

For every edge with `d_e >= 1`, `ApplyFloorplan` rewrites the IR:

1. Buffer edge: convert the on-chip ping-pong buffer to a `dataflow.stream` channel. If the consumer's access pattern is not streamable (random access), keep the buffer with the producer and insert a copy node on the consumer side (buffer -> stream -> buffer); the copy node is cheap and the cut edge is now a stream.
2. Stream edge: set depth to `max(current, 2 + 2*d_e + slack)`, where `2*d_e` is two register stages per boundary (AutoBridge/RapidStream rule) and slack is 2. Emit `#pragma HLS stream depth=N`.
3. Token streams: same depth rule.
4. Registering: for an internal `hls::stream` between two dataflow processes there is no C++ pragma that inserts a register stage on the crossing wires; the FIFO instance (`<name>_U`) is a single cell that Vivado places at one end. So the Tcl (a) pins the FIFO instance to the consumer's SLR, so the write side (data, write, full_n) is the crossing and the read side is local, and (b) marks the crossing nets `USER_CROSSING_SLR` so the placer routes them through Laguna TX/RX flops. The extra depth from step 2 keeps the producer from stalling on the longer round trip. This is the "C++-only" tier.
5. Optional RTL tier (phase 3 of the plan): replace the FIFO instance on each cut edge with AutoBridge's `relay_station` (almost-full FIFO with `LEVEL = d_e`) and insert registers on `ap_start`/`ap_done`/`start_for_*` sync signals of cut process pairs, exactly as TAPA's `add_pipeline` step does. This tier is what produced most of AutoBridge's gain. It is a post-HLS Verilog rewrite using the manifest, and should be implemented only after the C++ tier is measured.

Reconvergent paths: after depths are set, run AutoBridge's latency-balancing ILP (`t_src >= t_dst + L_e`, minimize `sum w_e (t_src - t_dst)`) and add the computed extra depth to the shallower branch so throughput is unchanged. Feedback cycles are protected by the cycle budget constraint above.

### 6.4 Where the top-level control goes

The Vitis dataflow controller of `forward` (FSM, `ap_sync_*`, `start_for_*` FIFOs) is not a unit. It is small and connects to every unit. Leave it unconstrained in the C++ tier (Vivado places it centrally); in the RTL tier, its per-process sync signals get one register per boundary. The existing design already excludes `forward` from assignment; keep that.

## 7. Constraint emission (component D)

`floorplan.tcl`:

- One pblock per SLR built from clock-region ranges with the shell removed (`resize_pblock -add CLOCKREGION_X0Y0:CLOCKREGION_X7Y3`, `-remove` for shell CRs), not `-add {SLRn}`; `CONTAIN_ROUTING` off by default (it hurts crossing routes), `SNAPPING_MODE ON`.
- `add_cells_to_pblock` using `get_cells -hierarchical -regexp` with the manifest regexes.
- FIFO instances of cut edges assigned to the consumer side.
- `USER_CROSSING_SLR` on cut nets; `KEEP_HIERARCHY SOFT` on unit instances.
- A "soft" variant that emits `USER_SLR_ASSIGNMENT SLRn` instead of pblocks, used to compare hard and soft constraint modes in the evaluation.
- Reports at the end: `report_utilization -slr`, `report_slr_crossing -cells`, `report_design_analysis -timing -congestion`, `get_timing_paths -inter_slr data_crossing -max_paths 200 -nworst 1 -filter {SLACK < 0}` with start/end cell names written as CSV.

## 8. The iterative loop (components E, F)

The loop has a cheap inner iteration and an expensive outer iteration.

Iteration 0 (compiler-only): estimator areas, ILP, emit. Minutes.

Iteration 0.5 (HLS calibration, about 10 to 30 minutes): run `csynth_design`; parse `<top>_csynth.rpt` and `<unit>_csynth.rpt` for exact LUT/FF/DSP/BRAM/URAM per unit and the module hierarchy. Replace estimator areas with measured ones (this fixes the BRAM and FF problems directly). Record a per-unit scale factor `area_actual/area_estimated` and persist it per node signature so later compiles of the same model start calibrated. Re-run the ILP if any SLR now violates the cap or if the imbalance changed by more than 10 percent. This single step is expected to give most of the benefit of "iteration".

Outer iteration k (Vivado, 1 to 3 hours per run; reduced with incremental P&R):

1. Parse feedback:
   - per-SLR utilization and per-pblock utilization: actual `U_k^r`;
   - SLL usage per boundary;
   - failing inter-SLR paths: map start/end cells to units via the manifest (`get_property PARENT` climb or path-prefix match), group by unit pair, keep worst slack and count;
   - congestion windows: map to units by cell names.
2. Update the model:
   - edge penalty `pen_e += lambda * max(0, -slack_e)` for edges whose unit pair had failing crossing paths; after two iterations with the same failing pair, force `d_e = 0` (grouping) or escalate that edge to the RTL tier;
   - capacity cap `rho_k -= 0.05` for an SLR with congestion windows above level 5 or with WNS-critical paths confined inside it;
   - node locks: units not touched by any failing path keep their SLR (enables `read_checkpoint -incremental` with high reuse);
   - if a failing path is internal to a unit (not a crossing), the floorplan is not the cause; tag the unit for the compiler-level fix (lower unroll factor or deeper pipelining of that node) rather than moving it.
3. Re-solve, re-emit, and re-run P&R with `read_checkpoint -incremental <prev_routed.dcp>` when more than 90 percent of cells are unchanged, otherwise from scratch.
4. Stop when WNS >= 0 at the target period, or after N iterations (N = 4), or when two consecutive iterations change no assignment. Keep the best iteration by (WNS, then SLL count).

Noise handling: every reported configuration is run with three placer directives (Default, ExtraTimingOpt, SSI_SpreadLogic_high) and the median WNS is used. The spread seen in the existing data (9 ns between runs) makes this mandatory.

The loop driver is a Python CLI: `hida-floorplan run --model resnet18.mlir --device u280.json --period 4.0 --iters 4`. Every iteration's graph, assignment, Tcl, and parsed reports are stored under `runs/<model>/<iter>/` so the thesis can show convergence plots.

## 9. Related work and positioning

| Work | Where it acts | What it does | What this design takes or changes |
|---|---|---|---|
| AutoBridge (FPGA'21) | post-HLS, RTL | ILP min-cut over task graph on a slot grid, per-slot capacity, SLL budget, 2 register stages per crossing via almost-full FIFOs, latency balancing ILP; 147 -> 297 MHz avg | ILP form, SLL constraint, pipelining rule, latency balancing. Difference: we decide at IR level with compiler-controlled names and can change buffer types and granularity. |
| TAPA / RapidStream 1.0, 2.0, RapidStream IR (TRETS'23, FPGA'22, ICCAD'24) | post-HLS, split compilation, anchor registers, DFX islands | Parallel P&R per island, 5 to 7x compile-time, up to 1.3x fmax | Feedback-parsing commands, floorplan DSE over area limits. We do not do split compilation. |
| PASTA (FCCM'23) | TAPA buffer channels | co-locates producer/consumer of BRAM channels | Same grouping rule for on-chip buffers. |
| FADO 1.0/2.0 (FPGA'23, TRETS'24) | directive + floorplan co-search | multi-dimensional bin packing, incremental legalization | Our outer loop can change HIDA parallel factors when the fix is not a move. |
| StreamTensor (MICRO'25, Chen lab) | compiler for LLM kernels | ILP die assignment (one-hot die variables; comm + imbalance objective) | Closest precedent; ours is generic HIDA, adds crossing treatment and feedback. |
| Prometheus (2025), ALDP (2025), FrqBooster (TCAD) | HLS-level SLR task partitioning | ILP partition with per-SLR DSP budgets; iterative congestion-driven refinement; +73/81 percent fmax | Validates that SLR-level partition of HLS tasks gives large gains on big designs. |
| Zheng et al. FPGA'14, ISDC DATE'24 | physically aware HLS | post-P&R delays fed back into scheduling/binding | Same spirit; we feed back into partitioning instead. |
| Josipovic et al. (FPGA'20/TRETS'22) | dataflow buffer placement MILP | throughput bound by cycle token ratio | Cycle budget constraint in Section 6.2. |

Gap: none of the above feeds Vivado per-SLR utilization, SLL usage and failing-crossing-path data back into an HLS compiler's node assignment with incremental P&R. That is the thesis claim.

## 10. Evaluation plan

Devices: pick one and keep it: U280 (xcvu37p, 3 SLRs, HBM in SLR0) or U250 (xcu250, 4 SLRs, DDR per SLR). The repo currently mixes the two.

Benchmarks: ResNet18, VGG16, MobileNet, YOLO from `samples/pytorch` at unroll/tile settings that reach 40 to 70 percent of the device; plus one or two PolyBench kernels compiled through the C++ front-end to show generality.

Baselines: (B0) vanilla HIDA, no constraints; (B1) HIDA + Vivado `SSI_*` placer directives only; (B2) HIDA + `USER_SLR_ASSIGNMENT` from the ILP, no crossing treatment; (B3) full C++ tier; (B4) plus RTL relay tier; (B5) iterations 1..N. If feasible, (B6) the same model expressed in TAPA with AutoBridge for an external reference.

Metrics: achieved Fmax (binary search on period, median of 3 directives), WNS at a fixed period, SLL usage per boundary, number of crossing nets, per-SLR utilization imbalance, throughput (frames/s = Fmax / II), compile time per iteration, and the number of iterations to closure.

Expected outcome: B0 fails or drops sharply in Fmax once the design spans SLRs; B2 alone may not help (the current data shows exactly that); B3 plus calibration should give most of the gain; B4 recovers the rest; iterations should mainly reduce variance and fix a few bad pairs.

## 11. Work plan

1. Weeks 1-2: fix the graph (shared-operand edges, hierarchy, anchors), device JSON, graph export. Unit tests with lit on small dataflow IR.
2. Weeks 3-4: stable naming, `inline off`, manifest, verify step. Confirm 100 percent cell match on ResNet18.
3. Weeks 5-6: ILP floorplanner in Python (CP-SAT), latency balancing, ApplyFloorplan (stream conversion, depths). Compare against BFS+FM on the same graphs.
4. Weeks 7-8: scale benchmarks up, run B0 to B3 with three directives each; establish that the problem exists at scale.
5. Weeks 9-10: feedback parser and outer loop with incremental P&R; convergence experiments.
6. Weeks 11-12: RTL relay tier (optional), final experiments, writing.

## 12. Risks

- Small designs cannot be improved by floorplanning. Mitigation: evaluate at scale (Section 10); report the size threshold where floorplanning starts to pay.
- Vivado result variance. Mitigation: multiple directives, medians, report distributions.
- Vitis HLS version drift in instance naming. Mitigation: manifest plus verify step; regexes anchored on the uid.
- Nodes larger than an SLR. Mitigation: unit splitting along buffer edges; otherwise reduce the node's unroll factor through the existing HIDA options and report it.
- Control-path crossings (`ap_sync`, `loop_dataflow_output_count`) that only the RTL tier can register. Mitigation: measure how much the C++ tier leaves on the table; implement the RTL tier if it is more than about 10 percent of Fmax.

## 14. Implementation status and workload notes

Implemented on branch `claude/fpga-floorplan-impl` (see `tools/floorplan/README.md`):

- `scalehls-export-dataflow-graph` and `scalehls-apply-floorplan` passes (components B and D), pipeline options `floorplan-graph-output` and `floorplan-assignment`, `#pragma HLS inline off` and `#pragma HLS stream depth` emission.
- `tools/floorplan/floorplan.py` with `solve` (CP-SAT ILP of Section 6.2, including anchors, grouping, cycle budget, locks and penalties), `emit` (pblock or soft constraints plus manifest), `calibrate` (csynth instance table) and `loop` (the outer iteration of Section 8), plus `parse_reports.py`, `run_vivado_iter.tcl`, `hls_csynth.tcl` and device files for U280 and U250.
- Not implemented yet: buffer-to-stream conversion of cut on-chip buffer edges (the solver keeps such pairs grouped instead), latency balancing of reconvergent paths, and the RTL relay-station tier.

HIDA's estimator reports LUT = 0 for every function and over-estimates BRAM, so the ILP balances on DSP until the first csynth calibration replaces the areas. Calibration is therefore the first step of every real run, not an optimization.

Transformer (LLM) workloads: a complete LLM is not a suitable benchmark because weights do not fit on chip and the PyTorch front-end requires static shapes, but a single encoder or decoder layer with a fixed sequence length (for example BERT-base dimensions at sequence length 128) is a good secondary workload. Its attention and feed-forward matmuls form 10 to 15 large units that unroll well and fill SLRs evenly, which exercises resource balancing more than CNNs do, and it allows a direct comparison with StreamTensor's ILP die assignment. Softmax and layer normalization need `exp`, `div` and `sqrt` operators that the estimator models only through the latency table in `config.json`, so the calibration step matters even more there. Keep the CNN benchmarks as the primary set, since that is where HIDA's own results are established.

## 13. References

- AutoBridge, FPGA 2021: https://dl.acm.org/doi/10.1145/3431920.3439289 ; code https://github.com/Licheng-Guo/AutoBridge (ILP in `src/autobridge/Floorplan/`, SLL constraints in `ILPUtilities.py`, latency balancing in `Route/latency_balancing.py`, 2 stages per boundary in `Opt/GlobalRouting.py`)
- TAPA, TRETS 2023: https://dl.acm.org/doi/10.1145/3609335 ; https://github.com/UCLA-VAST/tapa ; https://github.com/rapidstream-org/rapidstream-tapa
- RapidStream 1.0, FPGA 2022: https://dl.acm.org/doi/10.1145/3490422.3502361 ; RapidStream 2.0, TRETS 2023: https://dl.acm.org/doi/10.1145/3593025 ; RapidStream IR, ICCAD 2024: https://arxiv.org/abs/2410.13079
- PASTA, FCCM 2023 / TECS 2025: https://dl.acm.org/doi/10.1145/3676849
- TAPA-CS, ASPLOS 2024: https://dl.acm.org/doi/abs/10.1145/3620666.3651347
- FADO, FPGA 2023: https://dl.acm.org/doi/10.1145/3543622.3573188 ; FADO 2.0, TRETS 2024: https://dl.acm.org/doi/10.1145/3653458
- HIDA, ASPLOS 2024: https://arxiv.org/abs/2311.03379 ; StreamTensor, MICRO 2025: https://arxiv.org/abs/2509.13694
- Prometheus (holistic FPGA accelerator optimization, SLR-aware scheduling), 2025: https://arxiv.org/abs/2501.09242
- ALDP, algorithmic-level design partitioning for multi-die, 2025: https://doi.org/10.1145/3727646
- FrqBooster / congestion-aware floorplanning on multi-die FPGA: https://pure.ecnu.edu.cn/en/publications/frequency-oriented-hls-accelerator-exploration-via-congestion-awa/
- Josipovic et al., buffer placement and sizing for dataflow circuits, TRETS 2022: https://dl.acm.org/doi/10.1145/3477053
- Carloni and Sangiovanni-Vincentelli, latency-insensitive performance analysis, DAC 2000: https://dl.acm.org/doi/10.1145/337292.337441
- Zheng et al., P&R-directed HLS, FPGA 2014: https://dl.acm.org/doi/10.1145/2554688.2554775 ; ISDC, DATE 2024: https://arxiv.org/abs/2401.12343
- REFINE, FPGA 2024: https://dl.acm.org/doi/10.1145/3626202.3637560
- Vivado: UG949 hard SLR floorplan constraints https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/Using-Hard-SLR-Floorplan-Constraints ; `report_slr_crossing` https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_slr_crossing ; `get_timing_paths -inter_slr` https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_timing_paths ; `USER_SLR_ASSIGNMENT` / `USER_CROSSING_SLR` https://docs.amd.com/r/en-US/ug912-vivado-properties/USER_CROSSING_SLR ; incremental implementation https://docs.amd.com/r/2022.2-English/ug904-vivado-implementation/Incremental-Implementation-Controls
- Vitis HLS: inline pragma https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-inline ; stream directive https://docs.amd.com/r/en-US/ug1399-vitis-hls/syn.directive.stream ; Vitis SLR assignment https://xilinx.github.io/Vitis_Accel_Examples/2022.1/html/slr_assign.html
