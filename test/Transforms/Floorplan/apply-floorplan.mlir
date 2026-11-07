// RUN: scalehls-opt -scalehls-export-dataflow-graph="output=%t.json top-func=forward" -scalehls-apply-floorplan="assignment=%S/assignment.json" %s | FileCheck %s
// RUN: scalehls-opt -scalehls-export-dataflow-graph="output=%t.json top-func=forward" -scalehls-apply-floorplan="assignment=%S/assignment.json" %s | scalehls-translate -scalehls-emit-hlscpp | FileCheck --check-prefix=CPP %s

// CHECK: func.func @forward_node0({{.*}}) attributes {floorplan_unit, inline_off, slr = 0 : i32}
// CHECK: func.func @forward_node1({{.*}}) attributes {floorplan_unit, inline_off, slr = 1 : i32}
// CHECK: %0 = hls.dataflow.stream {depth = 2 : i32, fifo_depth = 6 : i32, floorplan_id = "forward:0"} : {{.*}}<i1, 2>

// CPP: void forward_node0(
// CPP: #pragma HLS inline off
// CPP: void forward_node1(
// CPP: #pragma HLS inline off
// CPP: void forward(
// CPP: hls::stream<bool> [[S:v[0-9]+]];
// CPP-NEXT: #pragma HLS stream variable=[[S]] depth=6

module {
  func.func @forward_node0(%arg0: !hls.stream<i1, 2>) {
    %true = arith.constant true
    hls.dataflow.stream_write %arg0, %true : !hls.stream<i1, 2>, i1
    return
  }
  func.func @forward_node1(%arg0: !hls.stream<i1, 2>) {
    hls.dataflow.stream_read %arg0 : (!hls.stream<i1, 2>) -> ()
    return
  }
  func.func @forward() attributes {top_func} {
    %0 = hls.dataflow.stream {depth = 2 : i32} : !hls.stream<i1, 2>
    func.call @forward_node0(%0) : (!hls.stream<i1, 2>) -> ()
    func.call @forward_node1(%0) : (!hls.stream<i1, 2>) -> ()
    return
  }
}
