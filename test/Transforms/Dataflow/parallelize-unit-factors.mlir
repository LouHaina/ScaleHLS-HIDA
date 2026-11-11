// RUN: scalehls-opt -scalehls-parallelize-dataflow-node="max-unroll-factor=4 record-unit-factors=true" %s | FileCheck %s --check-prefix=KNOB
// RUN: scalehls-opt -scalehls-parallelize-dataflow-node="max-unroll-factor=4 unit-factors=%S/Inputs/unit-factors.json" %s | FileCheck %s --check-prefix=UNIT
// RUN: not scalehls-opt -scalehls-parallelize-dataflow-node="max-unroll-factor=4 unit-factors=%S/Inputs/unit-factors-stale.json" %s 2>&1 | FileCheck %s --check-prefix=STALE

// Two top-level nodes with the same amount of computation. The plain knob gives
// both the same factor, which is recorded on the function.

// KNOB: func.func @forward({{.*}}) attributes {top_func, unit_unroll_factors = [5, 5]}
// KNOB: affine.for %{{.*}} = 0 to 16 step 8
// KNOB: affine.for %{{.*}} = 0 to 16 step 8

// With unit factors the first node is unrolled by 8 and the second by 2.

// UNIT: func.func @forward({{.*}}) attributes {top_func, unit_unroll_factors = [8, 2]}
// UNIT: hls.dataflow.node
// UNIT: affine.for %{{.*}} = 0 to 16 step 8
// UNIT: hls.dataflow.node
// UNIT: affine.for %{{.*}} = 0 to 16 step 2

// A factor file written for a different number of units is rejected.

// STALE: unit factors file lists 3 units but the function has 2 top-level dataflow nodes

func.func @forward(%arg0: memref<16xf32, #hls.mem<dram>>, %arg1: memref<16xf32, #hls.mem<bram_t2p>>, %arg2: memref<16xf32, #hls.mem<bram_t2p>>) attributes {top_func} {
  hls.dataflow.schedule(%arg0, %arg1, %arg2) : memref<16xf32, #hls.mem<dram>>, memref<16xf32, #hls.mem<bram_t2p>>, memref<16xf32, #hls.mem<bram_t2p>> {
  ^bb0(%arg3: memref<16xf32, #hls.mem<dram>>, %arg4: memref<16xf32, #hls.mem<bram_t2p>>, %arg5: memref<16xf32, #hls.mem<bram_t2p>>):
    hls.dataflow.node(%arg3) -> (%arg4) {inputTaps = [0 : i32], level = 1 : i32} : (memref<16xf32, #hls.mem<dram>>) -> memref<16xf32, #hls.mem<bram_t2p>> {
    ^bb0(%arg6: memref<16xf32, #hls.mem<dram>>, %arg7: memref<16xf32, #hls.mem<bram_t2p>>):
      affine.for %arg8 = 0 to 16 {
        %0 = affine.load %arg6[%arg8] : memref<16xf32, #hls.mem<dram>>
        %1 = arith.mulf %0, %0 : f32
        affine.store %1, %arg7[%arg8] : memref<16xf32, #hls.mem<bram_t2p>>
      } {parallel}
    }
    hls.dataflow.node(%arg4) -> (%arg5) {inputTaps = [0 : i32], level = 0 : i32} : (memref<16xf32, #hls.mem<bram_t2p>>) -> memref<16xf32, #hls.mem<bram_t2p>> {
    ^bb0(%arg6: memref<16xf32, #hls.mem<bram_t2p>>, %arg7: memref<16xf32, #hls.mem<bram_t2p>>):
      affine.for %arg8 = 0 to 16 {
        %0 = affine.load %arg6[%arg8] : memref<16xf32, #hls.mem<bram_t2p>>
        %1 = arith.mulf %0, %0 : f32
        affine.store %1, %arg7[%arg8] : memref<16xf32, #hls.mem<bram_t2p>>
      } {parallel}
    }
  }
  return
}
