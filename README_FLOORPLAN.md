# Vivado Floorplanning Workflow for HIDA Layout Partitioning

This document describes the complete workflow for generating Vivado-compatible floorplan constraints and running the full HLS synthesis + Vivado P&R flow.

## Overview

The workflow consists of three main steps:
1. **Generate floorplan constraints** using LayoutPartitionEnhanced pass
2. **Run HLS synthesis** to generate RTL
3. **Run Vivado P&R** with floorplan constraints applied

## Step 1: Generate Floorplan Constraints

Run the ScaleHLS optimization pipeline with layout partitioning enabled:

```bash
./build/bin/scalehls-opt ./samples/pytorch/resnet18/resnet18.mlir \
  --hida-pytorch-pipeline="top-func=forward \
                           loop-tile-size=4 \
                           loop-unroll-factor=2 \
                           enable-layout-partition=true \
                           num-tiles=4 \
                           partition-tcl-output=floorplan_constraints.tcl" \
  > output.mlir
```

### Parameters:
- `top-func=forward`: Top-level function name
- `loop-tile-size=4`: Loop tiling size
- `loop-unroll-factor=2`: Loop unroll factor
- `enable-layout-partition=true`: Enable layout partitioning
- `num-tiles=4`: Number of partitions (maps to 4 SLRs on U250)
- `partition-tcl-output=floorplan_constraints.tcl`: Output file for Vivado floorplan constraints

### Output:
- `floorplan_constraints.tcl`: Vivado-compatible floorplan constraints with:
  - Physical block (pblock) definitions for each partition
  - SLR assignments (SLR0, SLR1, SLR2, SLR3)
  - Resource region allocations (SLICE, DSP, BRAM)
  - Cell-to-pblock assignments based on function names
  - Inter-SLR routing optimizations

## Step 2: Generate HLS C++ Code

Convert the optimized MLIR to HLS C++ code:

```bash
./build/bin/scalehls-translate output.mlir \
  --emit-hlscpp \
  -o resnet18_8_4.cpp
```

### Output:
- `resnet18_8_4.cpp`: HLS C++ code ready for Vitis HLS synthesis

## Step 3: Run Complete HLS + Vivado P&R Flow

Use the provided `run_project.tcl` script to automate the entire flow:

```bash
vivado -mode batch -source run_project.tcl
```

### What `run_project.tcl` does:

1. **HLS Synthesis** (Vitis HLS):
   - Opens/creates HLS project `resnet18_8_4_partition`
   - Sets target device: `xcvu37p-fsvh2892-2L-e` (Alveo U250)
   - Adds source file: `resnet18_8_4.cpp`
   - Sets top function: `forward`
   - Applies clock constraints (5ns period, 0.2ns uncertainty)
   - Runs C synthesis (`csynth_design`)
   - Exports RTL design

2. **Vivado Project Setup**:
   - Creates Vivado project
   - Imports HLS-generated RTL
   - Applies timing constraints

3. **Synthesis**:
   - Runs Vivado synthesis
   - Generates post-synthesis checkpoint and reports

4. **Floorplan Application**:
   - Sources `floorplan_constraints.tcl`
   - Applies physical block constraints
   - Assigns cells to SLRs based on partition assignments

5. **Implementation (Place & Route)**:
   - Runs placement with SLR-aware optimization
   - Runs routing with inter-SLR crossing minimization
   - Performs post-route optimization

6. **Report Generation**:
   - Timing reports (timing_summary.rpt, timing_paths.rpt)
   - Utilization reports (utilization.rpt, pblock_utilization.rpt)
   - Power report (power.rpt)
   - Route status, DRC, clock networks, etc.

7. **Bitstream Generation**:
   - Generates bitstream: `resnet18_partition.bit`
   - Exports hardware platform: `resnet18_partition.xsa`

## Output Files

After running the complete flow, you will have:

### Checkpoints:
- `post_synth.dcp`: Post-synthesis design checkpoint
- `post_floorplan.dcp`: Post-floorplan design checkpoint
- `post_route.dcp`: Post-route design checkpoint

### Implementation Outputs:
- `resnet18_partition.bit`: FPGA bitstream
- `resnet18_partition.xsa`: Hardware platform for Vitis

### Reports:
- `synth_timing_summary.rpt`: Synthesis timing summary
- `synth_utilization.rpt`: Synthesis resource utilization
- `impl_timing_summary.rpt`: Implementation timing summary
- `impl_timing_paths.rpt`: Detailed timing paths
- `impl_utilization.rpt`: Implementation resource utilization
- `impl_pblock_utilization.rpt`: Per-partition resource utilization
- `impl_power.rpt`: Power analysis
- `route_status.rpt`: Routing status
- `drc.rpt`: Design rule check results
- `clock_networks.rpt`: Clock network analysis
- `high_fanout_nets.rpt`: High fanout net analysis

## Floorplan Constraint Structure

The generated `floorplan_constraints.tcl` contains:

### 1. Partition Resource Information (Comments)
```tcl
# Partition 0: LUT=0, FF=0, DSP=280, BRAM=28, URAM=0
# Partition 1: LUT=0, FF=0, DSP=184, BRAM=79, URAM=0
# Partition 2: LUT=100, FF=100, DSP=215, BRAM=82, URAM=0
# Partition 3: LUT=0, FF=0, DSP=172, BRAM=79, URAM=0
```

### 2. Physical Block Creation
```tcl
create_pblock pblock_partition_0
resize_pblock pblock_partition_0 -add {SLR0}
set_property SLR_ASSIGNMENT SLR0 [get_pblocks pblock_partition_0]
set_property CONTAIN_ROUTING true [get_pblocks pblock_partition_0]
```

### 3. Resource Region Allocation
```tcl
resize_pblock pblock_partition_0 -add {SLICE_X0Y0:SLICE_X107Y119}
resize_pblock pblock_partition_0 -add {DSP48E2_X0Y0:DSP48E2_X17Y47}
resize_pblock pblock_partition_0 -add {RAMB36_X0Y0:RAMB36_X7Y23 RAMB18_X0Y0:RAMB18_X15Y47}
```

### 4. Cell Assignment
```tcl
set partition_0_cells [list]
catch {lappend partition_0_cells [get_cells -hierarchical -filter "NAME =~ *forward_node156*"]}
catch {lappend partition_0_cells [get_cells -hierarchical -filter "NAME =~ *forward_node161*"]}
if {[llength $partition_0_cells] > 0} {
    add_cells_to_pblock pblock_partition_0 $partition_0_cells
}
```

### 5. Inter-SLR Optimization
```tcl
set_property ROUTE.SLR_CROSSING_MINIMIZATION true [current_design]
set_property ROUTE.SLR_CROSSING_OPTIMIZATION true [current_design]
set_property MAX_FANOUT 50 [get_nets -hierarchical]
```

## Device Information: Alveo U250

- **Part Number**: xcvu37p-fsvh2892-2L-e
- **Architecture**: Virtex UltraScale+ VU37P
- **SLRs**: 4 (SLR0, SLR1, SLR2, SLR3)
- **Total Resources**:
  - LUTs: ~864K
  - FFs: ~1728K
  - DSPs: ~6912
  - BRAMs: ~2496
  - URAMs: ~1280

### Per-SLR Resources (Approximate):
- LUTs: ~216K per SLR
- FFs: ~432K per SLR
- DSPs: ~1728 per SLR
- BRAMs: ~624 per SLR
- URAMs: ~320 per SLR

## Partition Strategy

The layout partitioning algorithm distributes functions across 4 partitions mapped to 4 SLRs:

- **Partition 0 → SLR0**: Control logic and top-level functions
- **Partition 1 → SLR1**: Early layer functions (high BRAM usage)
- **Partition 2 → SLR2**: Middle layer functions (highest resource usage)
- **Partition 3 → SLR3**: Late layer functions (balanced usage)

This distribution:
1. Balances resource utilization across SLRs
2. Minimizes inter-SLR communication overhead
3. Optimizes for timing closure
4. Reduces routing congestion

## Customization

### Changing Number of Partitions
Modify the `num-tiles` parameter:
```bash
--hida-pytorch-pipeline="... num-tiles=8 ..."
```

### Changing Output Filename
Modify the `partition-tcl-output` parameter:
```bash
--hida-pytorch-pipeline="... partition-tcl-output=my_floorplan.tcl ..."
```

### Modifying Floorplan Regions
Edit the generated `floorplan_constraints.tcl` to adjust:
- SLR assignments
- Resource region boundaries
- Cell assignment patterns

### Adjusting Timing Constraints
Edit `run_project.tcl` to modify:
- Clock period (default: 5.0ns = 200MHz)
- Clock uncertainty (default: 0.2ns)
- Max delay constraints

## Troubleshooting

### Issue: Timing violations
**Solution**: 
- Increase clock period in `run_project.tcl`
- Adjust floorplan regions to reduce inter-SLR crossings
- Enable more aggressive optimization strategies

### Issue: Resource overflow in a partition
**Solution**:
- Rerun with different `num-tiles` value
- Manually adjust pblock regions in `floorplan_constraints.tcl`
- Modify partition algorithm parameters

### Issue: Cells not assigned to pblocks
**Solution**:
- Check function naming patterns in `floorplan_constraints.tcl`
- Verify RTL hierarchy matches expected patterns
- Add additional wildcard patterns for cell matching

## References

- [Vivado Design Suite User Guide: Using Constraints (UG903)](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2021_1/ug903-vivado-using-constraints.pdf)
- [Vivado Design Suite User Guide: Implementation (UG904)](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2021_1/ug904-vivado-implementation.pdf)
- [UltraScale Architecture and Product Data Sheet (DS890)](https://www.xilinx.com/support/documentation/data_sheets/ds890-ultrascale-overview.pdf)
- [Alveo U250 Data Center Accelerator Card](https://www.xilinx.com/products/boards-and-kits/alveo/u250.html)

## Complete Command Sequence

```bash
# Step 1: Generate floorplan constraints
./build/bin/scalehls-opt ./samples/pytorch/resnet18/resnet18.mlir \
  --hida-pytorch-pipeline="top-func=forward loop-tile-size=4 loop-unroll-factor=2 enable-layout-partition=true num-tiles=4 partition-tcl-output=floorplan_constraints.tcl" \
  > output.mlir

# Step 2: Generate HLS C++ code
./build/bin/scalehls-translate output.mlir --emit-hlscpp -o resnet18_8_4.cpp

# Step 3: Run complete HLS + Vivado P&R flow
vivado -mode batch -source run_project.tcl

# Step 4: Check results
cat impl_timing_summary.rpt
cat impl_pblock_utilization.rpt
```

## Notes

- The floorplan constraints are applied **after** synthesis and **before** placement
- Inter-SLR routing is automatically optimized by Vivado when SLR assignments are specified
- The partition algorithm considers both resource utilization and communication patterns
- For best results, ensure HLS synthesis completes successfully before running Vivado P&R