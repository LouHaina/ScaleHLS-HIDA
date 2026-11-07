# One P&R iteration for the floorplan loop. Run with:
#   vivado -mode batch -source run_vivado_iter.tcl -tclargs <in.dcp> <floorplan.tcl> <period_ns> <out_dir> [<prev_routed.dcp>]
# Writes machine-readable reports that tools/floorplan/parse_reports.py understands.
set in_dcp     [lindex $argv 0]
set fp_tcl     [lindex $argv 1]
set period     [lindex $argv 2]
set out_dir    [lindex $argv 3]
set prev_dcp   [expr {[llength $argv] > 4 ? [lindex $argv 4] : ""}]
file mkdir $out_dir

open_checkpoint $in_dcp
source $fp_tcl
create_clock -period $period [get_ports ap_clk]
set_clock_uncertainty 0.2 [get_clocks ap_clk]

if {$prev_dcp ne "" && [file exists $prev_dcp]} {
    # Reuse the previous iteration's placement for unchanged cells.
    read_checkpoint -incremental $prev_dcp
}
opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force $out_dir/post_route.dcp
report_utilization -slr -file $out_dir/utilization_slr.rpt
foreach pb [get_pblocks] { report_utilization -pblocks $pb -file $out_dir/util_$pb.rpt }
report_timing_summary -file $out_dir/timing_summary.rpt
report_design_analysis -congestion -file $out_dir/congestion.rpt
report_slr_crossing -file $out_dir/slr_crossing.rpt
report_route_status -file $out_dir/route_status.rpt
if {$prev_dcp ne ""} { report_incremental_reuse -file $out_dir/incremental_reuse.rpt }

# Failing inter-SLR paths as CSV: slack, startpoint cell, endpoint cell.
set f [open $out_dir/crossing_paths.csv w]
puts $f "slack,source,destination"
foreach p [get_timing_paths -inter_slr data_crossing -setup -max_paths 500 -nworst 1 -filter {SLACK < 0}] {
    set s [get_property SLACK $p]
    set sp [get_cells -of_objects [get_property STARTPOINT_PIN $p]]
    set ep [get_cells -of_objects [get_property ENDPOINT_PIN $p]]
    puts $f "$s,$sp,$ep"
}
close $f
puts "ITERATION_DONE $out_dir WNS=[get_property SLACK [get_timing_paths -max_paths 1 -setup]]"
exit
