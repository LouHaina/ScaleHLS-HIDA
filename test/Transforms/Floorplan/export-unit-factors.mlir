// RUN: scalehls-opt -scalehls-export-dataflow-graph="output=%t.json top-func=forward" %s > /dev/null
// RUN: FileCheck --check-prefix=JSON --input-file=%t.json %s
// RUN: scalehls-opt -scalehls-export-dataflow-graph="output=%t.stale.json top-func=stale" %s 2>&1 > /dev/null | FileCheck --check-prefix=WARN %s
// RUN: FileCheck --check-prefix=STALE --input-file=%t.stale.json %s

// The unroll factors recorded by scalehls-parallelize-dataflow-node on the top
// function are exported per unit, in the program order of the units.

// JSON: "name": "forward_node0"
// JSON: "unit_index": 0
// JSON: "unroll": 8
// JSON: "name": "forward_node1"
// JSON: "unit_index": 1
// JSON: "unroll": 4

// If the number of units changed after the factors were recorded, they are not
// exported: applying them by position would hit the wrong units.

// WARN: recorded 3 unit unroll factors but found 2 units
// STALE: "name": "forward_node0"
// STALE: "unit_index": 0
// STALE: "unroll": -1
// STALE: "name": "forward_node1"
// STALE: "unit_index": 1
// STALE: "unroll": -1

module {
  func.func @forward_node0(%arg0: memref<64xi8, #hls.mem<dram>>) attributes {resource = #hls.res<lut = 0, dsp = 2, bram = 1>} {
    return
  }
  func.func @forward_node1(%arg0: memref<64xi8, #hls.mem<dram>>) attributes {resource = #hls.res<lut = 0, dsp = 4, bram = 0>} {
    return
  }
  func.func @forward(%arg0: memref<64xi8, #hls.mem<dram>>) attributes {unit_unroll_factors = [8, 4]} {
    func.call @forward_node0(%arg0) : (memref<64xi8, #hls.mem<dram>>) -> ()
    func.call @forward_node1(%arg0) : (memref<64xi8, #hls.mem<dram>>) -> ()
    return
  }
  func.func @stale(%arg0: memref<64xi8, #hls.mem<dram>>) attributes {unit_unroll_factors = [8, 4, 2]} {
    func.call @forward_node0(%arg0) : (memref<64xi8, #hls.mem<dram>>) -> ()
    func.call @forward_node1(%arg0) : (memref<64xi8, #hls.mem<dram>>) -> ()
    return
  }
}
