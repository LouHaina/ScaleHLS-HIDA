# Enhanced Layout Partition Pass for HIDA

## Overview

The Enhanced Layout Partition Pass is a resource-aware partitioning optimization pass for HIDA, an HLS compiler targeting Vitis/Vivado HLS. This pass partitions functions across multiple FPGA tiles to balance resource usage and minimize inter-tile communication.

## Features

- **Resource-Aware Partitioning**: Collects hardware resource usage (LUT, FF, DSP, BRAM, URAM) from QoREstimationPass
- **Balanced Algorithm**: Applies constraint-based partitioning to balance resources across tiles
- **Communication Optimization**: Minimizes inter-partition communication by analyzing function call graphs
- **TCL Generation**: Generates Vitis HLS TCL scripts for synthesis with partition directives
- **Configurable Parameters**: Supports various configuration options for different optimization strategies

## Architecture

```
Enhanced LayoutPartitionPass
├── Resource Collection Phase
│   ├── Collect QoR Resource Attributes
│   ├── Build Function Dependency Graph
│   └── Calculate Communication Costs
├── Partitioning Algorithm Phase
│   ├── Balanced Resource-Aare Algorithm
│   ├── Constraint Satisfaction
│   └── Communication Minimization
├── Annotation Phase
│   ├── Annotate Functions with Partition IDs
│   └── Set HLS Attributes
└── TCL Generation Phase
    ├── Generate Vitis HLS TCL Script
    └── Partition Assignment Directives
```

## Usage

### Command Line Options

```bash
scalehls-opt input.mlir -scalehls-qor-estimation -scalehls-layout-partition="tiles=4 tcl-output=partition.tcl balance-threshold=0.3 enable-comm-opt=true"
```

### Options

- `--tiles=<N>`: Number of FPGA tiles/partitions (default: 2)
- `--tcl-output=<path>`: Output path for TCL script (default: "partition.tcl")
- `--balance-threshold=<float>`: Resource balance threshold 0.0-1.0 (default: 0.3)
- `--enable-comm-opt=<bool>`: Enable communication optimization (default: true)

### Integration with Pipeline

The pass should be run after QoREstimationPass to collect resource information:

```cpp
// In pipeline configuration
pm.addPass(scalehls::createQoREstimationPass(targetSpec));
pm.addPass(scalehls::createLayoutPartitionPass());
```

### HIDA PyTorch Pipeline Integration

The layout partition pass is now fully integrated into the HIDA PyTorch pipeline and can be enabled with command-line options:

```bash
# Enable layout partitioning in the PyTorch pipeline
scalehls-opt resnet18.mlir -hida-pytorch-pipeline="top-func=forward loop-tile-size=8 loop-unroll-factor=4 enable-layout-partition=true"

# With custom partition settings
scalehls-opt resnet18.mlir -hida-pytorch-pipeline="top-func=forward loop-tile-size=8 loop-unroll-factor=4 enable-layout-partition=true num-tiles=8 partition-tcl-output=resnet18_partition.tcl partition-balance-threshold=0.2"

# Complete pipeline with HLS C++ generation
scalehls-opt resnet18.mlir -hida-pytorch-pipeline="top-func=forward loop-tile-size=8 loop-unroll-factor=4 enable-layout-partition=true num-tiles=4" | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives > resnet18.cpp
```

#### Pipeline Options

- `enable-layout-partition=<bool>`: Enable layout-driven partitioning (default: false)
- `num-tiles=<N>`: Number of FPGA tiles for partitioning (default: 4)
- `partition-tcl-output=<path>`: Output path for partition TCL script (default: "partition.tcl")
- `partition-balance-threshold=<float>`: Resource balance threshold (default: 0.3)

**Note**: In the integrated pipeline, QoR estimation is automatically run before layout partitioning to collect resource usage data.

## Implementation Details

### Resource Collection

The pass collects hardware resource information from function attributes set by QoREstimationPass:

```cpp
if (auto resourceAttr = getResource(func)) {
  info.resource.lut = resourceAttr.getLut();
  info.resource.dsp = resourceAttr.getDsp();
  info.resource.bram = resourceAttr.getBram();
  // Additional resource types...
}
```

### Partitioning Algorithm

The enhanced greedy algorithm considers:

1. **Resource Constraints**: Ensures each partition doesn't exceed resource limits
2. **Load Balancing**: Distributes resources evenly across partitions
3. **Communication Cost**: Minimizes inter-partition function calls

```cpp
// Scoring function
double resourceScore = utilizationRatio;
double communicationScore = (double)communicationCost / 10.0;
double totalScore = resourceScore + communicationScore;
```

### Function Annotation

Functions are annotated with partition IDs as integer attributes:

```mlir
func.func @compute_heavy(...) attributes {partition = 0 : i32} {
  // Function body
}
```

### TCL Generation

The pass generates Vitis HLS TCL scripts with partition directives:

```tcl
# Generated TCL script for HIDA Layout Partitioning
# Number of partitions: 4

# Partition 0 functions:
set_directive_allocation -limit 1 -type function compute_heavy -partition 0

# Partition 1 functions:
set_directive_allocation -limit 1 -type function compute_medium -partition 1

# Resource utilization per partition:
# Partition 0: LUT=5000, FF=5000, DSP=50, BRAM=20, URAM=0
# Partition 1: LUT=2500, FF=2500, DSP=25, BRAM=10, URAM=0
```

## Example

### Input MLIR

```mlir
module {
  func.func @compute_heavy(%arg0: memref<1024x1024xf32>) -> memref<1024x1024xf32> {
    // Heavy computation
    return %result : memref<1024x1024xf32>
  }
  
  func.func @compute_light(%arg0: memref<256x256xf32>) -> memref<256x256xf32> {
    // Light computation
    return %result : memref<256x256xf32>
  }
  
  func.func @main(%arg0: memref<1024x1024xf32>, %arg1: memref<256x256xf32>) {
    %0 = func.call @compute_heavy(%arg0) : (memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %1 = func.call @compute_light(%arg1) : (memref<256x256xf32>) -> memref<256x256xf32>
    return
  }
}
```

### Output MLIR (After Partitioning)

```mlir
module {
  func.func @compute_heavy(%arg0: memref<1024x1024xf32>) -> memref<1024x1024xf32> 
    attributes {partition = 0 : i32} {
    // Heavy computation
    return %result : memref<1024x1024xf32>
  }
  
  func.func @compute_light(%arg0: memref<256x256xf32>) -> memref<256x256xf32> 
    attributes {partition = 1 : i32} {
    // Light computation
    return %result : memref<256x256xf32>
  }
  
  func.func @main(%arg0: memref<1024x1024xf32>, %arg1: memref<256x256xf32>) 
    attributes {partition = 0 : i32} {
    %0 = func.call @compute_heavy(%arg0) : (memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %1 = func.call @compute_light(%arg1) : (memref<256x256xf32>) -> memref<256x256xf32>
    return
  }
}
```

## Testing

Run the test case:

```bash
scalehls-opt test/Transforms/layout-partition-test.mlir -scalehls-qor-estimation -scalehls-layout-partition="tiles=4 tcl-output=test_partition.tcl"
```

## Future Enhancements

1. **Advanced Algorithms**: Integration with METIS/hMETIS for more sophisticated partitioning
2. **Dynamic Resource Constraints**: Support for different resource limits per partition
3. **Multi-objective Optimization**: Balance between resource usage, communication, and timing
4. **Hierarchical Partitioning**: Support for nested partitioning strategies
5. **Machine Learning Integration**: Use ML models to predict optimal partitioning strategies

## Related Work

This implementation is inspired by the Python prototype in `demo.py` which uses OR-Tools for constraint-based optimization. The C++ implementation provides a more efficient and integrated solution within the MLIR infrastructure.

## References

- [ScaleHLS Documentation](https://github.com/hanchenye/scalehls)
- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [Vitis HLS User Guide](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2021_1/ug1399-vitis-hls.pdf)