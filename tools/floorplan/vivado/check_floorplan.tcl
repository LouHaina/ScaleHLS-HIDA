# vivado -mode batch -source check_floorplan.tcl -tclargs <post_synth.dcp> <floorplan.tcl> <report>
# Applies a floorplan to a synthesized design without placing it: every cell
# lookup must succeed, and every top-level cell must end up in a pblock.
# never truncate collections converted to Tcl lists
set_param tcl.collectionResultDisplayLimit 0
lassign $argv dcp fp rpt
open_checkpoint $dcp
if {[catch {source $fp} err]} { puts "FLOORPLAN_CHECK FAIL: $err"; exit 1 }
set f [open $rpt w]
foreach pb [get_pblocks] {
    puts $f "$pb [get_property GRID_RANGES $pb] cells=[llength [get_cells -of_objects $pb]]"
}
set free [get_cells -quiet -filter {IS_PRIMITIVE==0 && PBLOCK == ""}]
set prim [get_cells -filter {IS_PRIMITIVE==1}]
set prim_free [get_cells -quiet -filter {IS_PRIMITIVE==1 && PBLOCK == ""}]
puts $f "top_primitives=[llength $prim] unassigned_top_primitives=[llength $prim_free]"
if {[llength $prim_free] > 0} { puts "FLOORPLAN_CHECK FAIL: [llength $prim_free] top-level primitives not in a pblock"; close $f; exit 1 }
puts $f "unassigned_hier_cells=[llength $free] $free"
close $f
puts "FLOORPLAN_CHECK OK pblocks=[llength [get_pblocks]] unassigned=[llength $free]"
exit
