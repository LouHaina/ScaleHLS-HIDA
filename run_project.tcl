open_checkpoint post_floorplan_ooc.dcp
source floorplan_constraints.tcl
create_clock -period 8.0 [get_ports ap_clk]   ;# 125 MHz
set_clock_uncertainty 0.2 [get_clocks ap_clk]
opt_design
place_design
phys_opt_design
route_design

report_utilization -slr -file utilization_slr.rpt
foreach pb [get_pblocks] {
    report_utilization -pblocks $pb -file util_$pb.rpt
}
report_timing_summary -file timing_summary.rpt
report_design_analysis -congestion -file congestion.rpt
report_route_status -file route_status.rpt

exit
