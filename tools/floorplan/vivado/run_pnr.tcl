# Same flow as data/run_*/run.tcl and the PR's run_vivado_iter.tcl, plus a placer directive.
# vivado -mode batch -source run_one.tcl -tclargs <floorplan.tcl> <directive> <out_dir>
lassign $argv fp_tcl directive out_dir
file mkdir $out_dir
set t0 [clock seconds]
open_checkpoint [expr {[info exists ::env(FP_DCP)] ? $::env(FP_DCP) : "post_floorplan_ooc.dcp"}]
source $fp_tcl
create_clock -period 8.0 [get_ports ap_clk]
set_clock_uncertainty 0.2 [get_clocks ap_clk]
opt_design
place_design -directive $directive
phys_opt_design
route_design
write_checkpoint -force $out_dir/post_route.dcp
report_utilization -slr -file $out_dir/utilization_slr.rpt
foreach pb [get_pblocks] { report_utilization -pblocks $pb -file $out_dir/util_$pb.rpt }
report_timing_summary -file $out_dir/timing_summary.rpt
report_design_analysis -congestion -file $out_dir/congestion.rpt
report_route_status -file $out_dir/route_status.rpt
set f [open $out_dir/crossing_paths.csv w]
puts $f "slack,source,destination"
foreach p [get_timing_paths -inter_slr data_crossing -setup -max_paths 500 -nworst 1 -filter {SLACK < 0}] {
    puts $f "[get_property SLACK $p],[get_cells -of_objects [get_property STARTPOINT_PIN $p]],[get_cells -of_objects [get_property ENDPOINT_PIN $p]]"
}
close $f
set wp [get_timing_paths -max_paths 1 -setup]
set f [open $out_dir/summary.txt w]
puts $f "fp=$fp_tcl directive=$directive wns=[get_property SLACK $wp] runtime_s=[expr {[clock seconds]-$t0}]"
puts $f "worst_path=[get_property STARTPOINT_PIN $wp] -> [get_property ENDPOINT_PIN $wp]"
close $f
exit
