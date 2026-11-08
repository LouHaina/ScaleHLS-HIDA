# Out-of-context synthesis of the Vitis HLS RTL -> post_synth.dcp
#   vivado -mode batch -source synth_ooc.tcl -tclargs <cfg_dir>
set d [lindex $argv 0]
set v $d/hls_prj/sol/syn/verilog
create_project -in_memory -part xcvu37p-fsvh2892-2L-e
foreach ip [glob -nocomplain $v/*_ip.tcl] { source $ip }
if {[llength [get_ips]] > 0} { generate_target all [get_ips]; synth_ip [get_ips] }
read_verilog [glob $v/*.v]
foreach dat [glob -nocomplain $v/*.dat] { file copy -force $dat . }
synth_design -top forward -mode out_of_context -part xcvu37p-fsvh2892-2L-e
create_clock -period 3.33 [get_ports ap_clk]
write_checkpoint -force $d/post_synth.dcp
report_utilization -file $d/synth_util.rpt
report_utilization -hierarchical -hierarchical_depth 1 -file $d/hier_util.rpt
puts SYNTH_DONE
exit
