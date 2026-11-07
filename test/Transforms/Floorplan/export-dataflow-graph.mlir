// RUN: scalehls-opt -scalehls-export-dataflow-graph="output=%t.json top-func=forward" %s | FileCheck %s
// RUN: FileCheck --check-prefix=JSON --input-file=%t.json %s

// CHECK: func.func @forward_node0_leaf(%arg0: memref<16xi8, #hls.mem<bram_t2p>>) attributes {inline, resource = #hls.res<lut = 0, dsp = 2, bram = 0>}
// CHECK: func.func @forward_node0({{.*}}) attributes {floorplan_unit, func_directive = #hls.func<pipeline = false, target_interval = 1, dataflow = true>, inline_off, resource = #hls.res<lut = 0, dsp = 2, bram = 1>} {
// CHECK:   %0 = hls.dataflow.buffer {depth = 1 : i32, floorplan_id = "forward_node0:{{[0-9]+}}"} : memref<16xi8, #hls.mem<bram_t2p>>
// CHECK: func.func @forward_node1({{.*}}) attributes {floorplan_unit, inline_off, resource = #hls.res<lut = 0, dsp = 4, bram = 0>}
// CHECK: func.func @forward({{.*}}) attributes {top_func} {
// CHECK:   %0 = hls.dataflow.stream {depth = 2 : i32, floorplan_id = "forward:{{[0-9]+}}"} : {{.*}}<i1, 2>

// JSON: "top": "forward"
// JSON: "name": "forward_node0"
// JSON: "is_unit": true
// JSON: "name": "forward_node0_leaf"
// JSON: "parent": "forward_node0"
// JSON: "is_unit": false
// JSON: "name": "forward_node1"
// JSON: "is_unit": true
// JSON: "edges"
// JSON: "src": "forward_node0_leaf"
// JSON: "dst": "forward_node0_leaf2"
// JSON: "kind": "buffer"
// JSON: "src": "forward_node0"
// JSON: "dst": "forward_node1"
// JSON: "kind": "token"
// JSON: "src": "forward_node0"
// JSON: "dst": "forward_node1"
// JSON: "kind": "port"

module {
  func.func @forward_node0_leaf(%arg0: memref<16xi8, #hls.mem<bram_t2p>>) attributes {inline, resource = #hls.res<lut = 0, dsp = 2, bram = 0>} {
    return
  }
  func.func @forward_node0_leaf2(%arg0: memref<16xi8, #hls.mem<bram_t2p>>, %arg1: memref<64xi8, #hls.mem<dram>>) attributes {inline} {
    return
  }
  func.func @forward_node0(%arg0: !hls.stream<i1, 2>, %arg1: memref<64xi8, #hls.mem<dram>>) attributes {func_directive = #hls.func<pipeline = false, target_interval = 1, dataflow = true>, resource = #hls.res<lut = 0, dsp = 2, bram = 1>} {
    %0 = hls.dataflow.buffer {depth = 1 : i32} : memref<16xi8, #hls.mem<bram_t2p>>
    func.call @forward_node0_leaf(%0) : (memref<16xi8, #hls.mem<bram_t2p>>) -> ()
    func.call @forward_node0_leaf2(%0, %arg1) : (memref<16xi8, #hls.mem<bram_t2p>>, memref<64xi8, #hls.mem<dram>>) -> ()
    %true = arith.constant true
    hls.dataflow.stream_write %arg0, %true : !hls.stream<i1, 2>, i1
    return
  }
  func.func @forward_node1(%arg0: !hls.stream<i1, 2>, %arg1: memref<64xi8, #hls.mem<dram>>) attributes {resource = #hls.res<lut = 0, dsp = 4, bram = 0>} {
    hls.dataflow.stream_read %arg0 : (!hls.stream<i1, 2>) -> ()
    return
  }
  func.func @forward(%arg0: memref<64xi8, #hls.mem<dram>>) attributes {top_func} {
    %0 = hls.dataflow.stream {depth = 2 : i32} : !hls.stream<i1, 2>
    func.call @forward_node0(%0, %arg0) : (!hls.stream<i1, 2>, memref<64xi8, #hls.mem<dram>>) -> ()
    func.call @forward_node1(%0, %arg0) : (!hls.stream<i1, 2>, memref<64xi8, #hls.mem<dram>>) -> ()
    return
  }
}
