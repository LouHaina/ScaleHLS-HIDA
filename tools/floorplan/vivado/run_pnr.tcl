# vivado -mode batch -source run_pnr.tcl -tclargs <cfg_dir> <floorplan.tcl|none> <directive> <out_dir> <stage> <strip>
#   stage: place  -> opt/place/phys_opt, post_place.dcp, reports (fast screening)
#          route  -> also route_design, post_route.dcp, reports
#   strip: full | <x0>-<x1>  clock-region columns the whole design is confined to,
#          in every SLR (emulates a smaller multi-SLR device so a mid-size design
#          has to cross SLRs).  With fp=none the whole design goes into one strip
#          pblock; floorplan Tcl files are expected to use the same columns.
# never truncate collections converted to Tcl lists
set_param tcl.collectionResultDisplayLimit 0
lassign $argv cfg fp_tcl directive out_dir stage strip
if {$stage eq ""} { set stage route }
if {$strip eq ""} { set strip full }
set_param general.maxThreads 8
file mkdir $out_dir
set t0 [clock seconds]
if {$stage ne "resume"} {
    open_checkpoint $cfg/post_synth.dcp
    catch {delete_pblocks [get_pblocks]}
    if {$fp_tcl ne "none"} { source $fp_tcl }
    if {$strip ne "full"} {
        # everything the floorplan did not place (all cells for fp=none) stays in the strip
        lassign [split $strip -] x0 x1
        create_pblock pb_strip
        resize_pblock pb_strip -add "CLOCKREGION_X${x0}Y0:CLOCKREGION_X${x1}Y11"
        set_property SNAPPING_MODE ON [get_pblocks pb_strip]
        set rest [get_cells -quiet -filter {PBLOCK == ""}]
        if {[llength $rest]} { add_cells_to_pblock pb_strip $rest }
        puts "strip $strip: [llength $rest] top-level cells in pb_strip"
    }
    create_clock -period 3.33 [get_ports ap_clk]
    set_clock_uncertainty 0.2 [get_clocks ap_clk]
}

proc wns {} { get_property SLACK [get_timing_paths -max_paths 1 -setup] }
proc tns {} {
    # first data row of the Design Timing Summary: WNS TNS ...
    set r [report_timing_summary -no_detailed_paths -return_string]
    if {[regexp {WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[0-9.]+)\s+(-?[0-9.]+)} $r -> w t]} { return $t }
    return nan
}
proc dump {out_dir tag} {
    report_utilization -slr -file $out_dir/utilization_slr_$tag.rpt
    report_timing_summary -max_paths 20 -file $out_dir/timing_summary_$tag.rpt
    set f [open $out_dir/paths_$tag.csv w]
    puts $f "slack,levels,logic,route,inter_slr,start,end"
    foreach p [get_timing_paths -setup -max_paths 500 -nworst 1] {
        set sp [get_property STARTPOINT_PIN $p]; set ep [get_property ENDPOINT_PIN $p]
        set s1 [get_slrs -quiet -of_objects [get_cells -of_objects $sp]]
        set s2 [get_slrs -quiet -of_objects [get_cells -of_objects $ep]]
        set xs [expr {[llength $s1] && [llength $s2] && $s1 ne $s2}]
        puts $f "[get_property SLACK $p],[get_property LOGIC_LEVELS $p],[get_property DATAPATH_LOGIC_DELAY $p],[get_property DATAPATH_NET_DELAY $p],$xs,$sp,$ep"
    }
    close $f
    set f [open $out_dir/unit_place_$tag.csv w]
    puts $f "unit,slrs"
    foreach u [get_cells -filter {IS_PRIMITIVE==0}] {
        set sl [lsort -unique [get_slrs -quiet -of_objects [get_cells -hier -filter "IS_PRIMITIVE==1 && NAME =~ $u/*"]]]
        puts $f "$u,[join $sl {;}]"
    }
    close $f
}

if {$stage eq "resume"} {
    # route a previous place-stage run of the same out_dir
    set line [string trim [read [open $out_dir/summary.txt]]]
    open_checkpoint $out_dir/post_place.dcp
} else {
    opt_design
    place_design -directive $directive
    phys_opt_design
    write_checkpoint -force $out_dir/post_place.dcp
    set wp [wns]; set tp [tns]; set t_place [expr {[clock seconds]-$t0}]
    dump $out_dir place
    set line "fp=$fp_tcl directive=$directive strip=$strip place_wns=$wp place_tns=$tp place_s=$t_place"
}
if {$stage eq "route" || $stage eq "resume"} {
    route_design
    write_checkpoint -force $out_dir/post_route.dcp
    report_route_status -file $out_dir/route_status.rpt
    report_design_analysis -congestion -file $out_dir/congestion.rpt
    dump $out_dir route
    append line " route_wns=[wns] route_tns=[tns] total_s=[expr {[clock seconds]-$t0}]"
}
set f [open $out_dir/summary.txt w]; puts $f $line; close $f
exit
