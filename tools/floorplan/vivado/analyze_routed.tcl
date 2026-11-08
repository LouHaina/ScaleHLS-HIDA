# Feedback from a routed design: per-unit placement bbox (clock regions) and top paths.
#   vivado -mode batch -source analyze_routed.tcl -tclargs <routed.dcp> <out_dir>
lassign $argv dcp out_dir
file mkdir $out_dir
open_checkpoint $dcp
set f [open $out_dir/unit_place.csv w]
puts $f "unit,n_cells,crs,slrs"
foreach u [concat [get_cells -filter {IS_PRIMITIVE==0}] top] {
    if {$u eq "top"} { set prims [get_cells -filter {IS_PRIMITIVE==1}] } else {
        set prims [get_cells -hier -filter "IS_PRIMITIVE==1 && NAME =~ $u/* && (PRIMITIVE_LEVEL==LEAF || PRIMITIVE_LEVEL==MACRO)"] }
    set sites [get_sites -quiet -of_objects $prims]
    set crs [lsort -unique [get_clock_regions -quiet -of_objects $sites]]
    set slrs [lsort -unique [get_slrs -quiet -of_objects $sites]]
    puts $f "$u,[llength $prims],[join $crs {;}],[join $slrs {;}]"
}
close $f
set f [open $out_dir/paths.csv w]
puts $f "slack,levels,logic,route,start,end"
foreach p [get_timing_paths -setup -max_paths 300 -nworst 1] {
    puts $f "[get_property SLACK $p],[get_property LOGIC_LEVELS $p],[get_property DATAPATH_LOGIC_DELAY $p],[get_property DATAPATH_NET_DELAY $p],[get_property STARTPOINT_PIN $p],[get_property ENDPOINT_PIN $p]"
}
close $f
report_timing -max_paths 5 -nworst 1 -input_pins -file $out_dir/top5.rpt
puts ANALYZE_DONE
exit
