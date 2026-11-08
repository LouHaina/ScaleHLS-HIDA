# vivado -mode batch -source run_pnr.tcl -tclargs <cfg_dir> <floorplan.tcl|none> <directive> <out_dir>
lassign $argv cfg fp_tcl directive out_dir
file mkdir $out_dir
set t0 [clock seconds]
open_checkpoint $cfg/post_synth.dcp
catch {delete_pblocks [get_pblocks]}
if {$fp_tcl ne "none"} { source $fp_tcl }
create_clock -period 3.33 [get_ports ap_clk]
set_clock_uncertainty 0.2 [get_clocks ap_clk]
opt_design
place_design -directive $directive
phys_opt_design
route_design
write_checkpoint -force $out_dir/post_route.dcp
report_utilization -slr -file $out_dir/utilization_slr.rpt
report_timing_summary -file $out_dir/timing_summary.rpt
report_design_analysis -congestion -file $out_dir/congestion.rpt
report_route_status -file $out_dir/route_status.rpt
set f [open $out_dir/paths.csv w]
puts $f "slack,levels,logic,route,inter_slr,start,end"
foreach p [get_timing_paths -setup -max_paths 500 -nworst 1] {
    set sp [get_property STARTPOINT_PIN $p]; set ep [get_property ENDPOINT_PIN $p]
    set xs [expr {[llength [get_slrs -quiet -of_objects [get_cells -of_objects $sp]]] && [llength [get_slrs -quiet -of_objects [get_cells -of_objects $ep]]] && [get_slrs -of_objects [get_cells -of_objects $sp]] ne [get_slrs -of_objects [get_cells -of_objects $ep]]}]
    puts $f "[get_property SLACK $p],[get_property LOGIC_LEVELS $p],[get_property DATAPATH_LOGIC_DELAY $p],[get_property DATAPATH_NET_DELAY $p],$xs,$sp,$ep"
}
close $f
set f [open $out_dir/unit_place.csv w]
puts $f "unit,slrs"
foreach u [get_cells -filter {IS_PRIMITIVE==0}] {
    set sl [lsort -unique [get_slrs -quiet -of_objects [get_cells -hier -filter "IS_PRIMITIVE==1 && NAME =~ $u/*"]]]
    puts $f "$u,[join $sl {;}]"
}
close $f
set wp [get_timing_paths -max_paths 1 -setup]
set f [open $out_dir/summary.txt w]
puts $f "fp=$fp_tcl directive=$directive wns=[get_property SLACK $wp] runtime_s=[expr {[clock seconds]-$t0}]"
close $f
exit
