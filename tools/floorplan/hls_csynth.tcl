# Vitis HLS: synthesize the HIDA output and produce a post-synthesis Vivado
# checkpoint for the floorplan loop. Usage:
#   vitis_hls -f hls_csynth.tcl -tclargs <design.cpp> <out_dir> [<top>] [<part>] [<period_ns>]
set cpp    [lindex $argv 0]
set out    [lindex $argv 1]
set top    [expr {[llength $argv] > 2 ? [lindex $argv 2] : "forward"}]
set part   [expr {[llength $argv] > 3 ? [lindex $argv 3] : "xcu280-fsvh2892-2L-e"}]
set period [expr {[llength $argv] > 4 ? [lindex $argv 4] : 4.0}]

open_project -reset $out/hls_prj
set_top $top
add_files $cpp
open_solution -reset sol1 -flow_target vivado
set_part $part
create_clock -period $period -name default
config_compile -name_max_length 80
config_rtl -register_all_io
csynth_design
export_design -format ip_catalog -rtl verilog
# The csynth report with the per-instance table used by floorplan.py calibrate:
file copy -force $out/hls_prj/sol1/syn/report/${top}_csynth.rpt $out/${top}_csynth.rpt
exit
