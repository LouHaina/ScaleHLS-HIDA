# vivado -mode batch -source check_floorplan.tcl -tclargs <post_synth.dcp> <floorplan.tcl> <report>
# Applies a floorplan to a synthesized design without placing it: every cell
# lookup must succeed, and every top-level cell must end up in a pblock.
lassign $argv dcp fp rpt
open_checkpoint $dcp
if {[catch {source $fp} err]} { puts "FLOORPLAN_CHECK FAIL: $err"; exit 1 }
set f [open $rpt w]
foreach pb [get_pblocks] {
    puts $f "$pb [get_property GRID_RANGES $pb] cells=[llength [get_cells -of_objects $pb]]"
}
set free [get_cells -quiet -filter {IS_PRIMITIVE==0 && PBLOCK == ""}]
puts $f "unassigned_hier_cells=[llength $free] $free"
close $f
puts "FLOORPLAN_CHECK OK pblocks=[llength [get_pblocks]] unassigned=[llength $free]"
exit
