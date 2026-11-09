// RUN: scalehls-opt -scalehls-physical-floorplan="device=%S/Inputs/tiny_device.csv out-tcl=%t.tcl out-json=%t.json" %s | FileCheck %s
// RUN: FileCheck %s --check-prefix=TCL < %t.tcl
// RUN: FileCheck %s --check-prefix=JSON0 < %t.json
// RUN: scalehls-opt -scalehls-physical-floorplan="device=%S/Inputs/tiny_device.csv feedback=%S/Inputs/feedback.json out-tcl=%t.fb.tcl out-json=%t.fb.json" %s -o /dev/null
// RUN: FileCheck %s --check-prefix=FB < %t.fb.json
// RUN: FileCheck %s --check-prefix=FBTCL < %t.fb.tcl

// Iteration 0: nodes, AXI bundles and the controller are placed from the IR.
// CHECK: func.func @forward({{.*}}hls.floorplan_slot = "SLR1"{{.*}}top_func
// CHECK: hls.axi.bundle "axi_0" {hls.floorplan_slot = "SLR{{[0-2]}}"}
// CHECK: call @node0({{.*}}hls.floorplan_slot = "SLR{{[0-2]}}"
// CHECK: call @node1({{.*}}hls.floorplan_slot = "SLR{{[0-2]}}"

// TCL: create_pblock pb_SLR
// TCL: resize_pblock pb_SLR{{[0-2]}} -add {CLOCKREGION_X0Y{{[0-9]+}}:CLOCKREGION_X0Y{{[0-9]+}}}
// TCL: get_cells -quiet {axi_0_m_axi_U}

// JSON0: "action": "initial"
// JSON0: "nodes": 5

// With feedback: measured cells replace the IR proxies, the unused bundle
// axi_1 has no netlist cell and is dropped, the previous assignment is kept,
// and the dominant path class (control broadcast) selects the RTL pass.
// FB: "action": "rtl:ctrl"
// FB: "node0": "SLR0"
// FB: "node1": "SLR1"
// FB: "dropped": [
// FB-NEXT: "axi_1"
// FB: "moved": 0
// FB: "rtl_passes": [
// FB-NEXT: "ctrl"

// FBTCL: get_cells -quiet {grp_node0_fu_10}
// FBTCL: get_cells -quiet {control_s_axi_U}

module {
  func.func @node0(%arg0: memref<16xf32, #hls.mem<dram>>, %arg1: memref<16xf32>) attributes {resource = #hls.res<lut = 0, dsp = 100, bram = 0>} {
    return
  }
  func.func @node1(%arg0: memref<16xf32>) attributes {resource = #hls.res<lut = 0, dsp = 100, bram = 0>} {
    return
  }
  func.func @forward(%arg0: !hls.axi<memref<16xf32, #hls.mem<dram>>>, %arg1: !hls.axi<memref<16xf32, #hls.mem<dram>>>) attributes {top_func} {
    %0 = hls.axi.bundle "axi_0" : <f32, mm>
    %1 = hls.axi.port %0, %arg0 : <f32, mm>, (!hls.axi<memref<16xf32, #hls.mem<dram>>>) -> memref<16xf32, #hls.mem<dram>>
    %2 = hls.axi.bundle "axi_1" : <f32, mm>
    %3 = hls.axi.port %2, %arg1 : <f32, mm>, (!hls.axi<memref<16xf32, #hls.mem<dram>>>) -> memref<16xf32, #hls.mem<dram>>
    %4 = memref.alloc() : memref<16xf32>
    call @node0(%1, %4) {resource = #hls.res<lut = 0, dsp = 100, bram = 0>} : (memref<16xf32, #hls.mem<dram>>, memref<16xf32>) -> ()
    call @node1(%4) {resource = #hls.res<lut = 0, dsp = 100, bram = 0>} : (memref<16xf32>) -> ()
    return
  }
}
