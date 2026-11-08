# Netlist feedback for the floorplanner (PR #4 doc, components B/E):
# measured per-unit area and the real unit-to-unit connectivity of a
# synthesized checkpoint, plus the device's per-clock-region capacity.
#   vivado -mode batch -source extract_graph.tcl -tclargs <in.dcp> <out_dir>
lassign $argv in_dcp out_dir
file mkdir $out_dir
open_checkpoint $in_dcp
catch {delete_pblocks [get_pblocks]}

# Units = top-level hierarchical cells; top-level primitives form "top".
proc unit_of {cell} {
    set n [lindex [split $cell /] 0]
    if {[get_property IS_PRIMITIVE [get_cells $n]]} { return "top" }
    return $n
}
set f [open $out_dir/units.csv w]
puts $f "unit,lut,ff,dsp,bram,uram,carry"
set tops [get_cells -filter {IS_PRIMITIVE==0}]
foreach u $tops {
    set prims [get_cells -hier -filter "IS_PRIMITIVE==1 && NAME =~ $u/*"]
    set lut [llength [filter $prims {PRIMITIVE_GROUP==LUT || PRIMITIVE_GROUP==LUTRAM || PRIMITIVE_GROUP==DMEM}]]
    set ff  [llength [filter $prims {PRIMITIVE_GROUP==FLOP_LATCH || PRIMITIVE_GROUP==REGISTER}]]
    set dsp [llength [filter $prims {PRIMITIVE_GROUP==ARITHMETIC}]]
    set br  [llength [filter $prims {PRIMITIVE_GROUP==BLOCKRAM}]]
    set ur  [llength [filter $prims {PRIMITIVE_TYPE =~ BLOCKRAM.URAM.*}]]
    set cy  [llength [filter $prims {PRIMITIVE_GROUP==CARRY}]]
    puts $f "$u,$lut,$ff,$dsp,$br,$ur,$cy"
}
set tp [get_cells -filter {IS_PRIMITIVE==1}]
puts $f "top,[llength [filter $tp {PRIMITIVE_GROUP==LUT}]],[llength [filter $tp {PRIMITIVE_GROUP==FLOP_LATCH || PRIMITIVE_GROUP==REGISTER}]],0,0,0,0"
close $f

# Top-level nets: which units each one touches (driver first), fanout.
set f [open $out_dir/nets.csv w]
puts $f "net,driver,sinks,fanout"
foreach n [get_nets -filter {TYPE != GROUND && TYPE != POWER}] {
    set dp [get_pins -quiet -of_objects $n -filter {DIRECTION==OUT}]
    if {[llength $dp] == 0} { continue }
    set drv [unit_of [get_cells -of_objects [lindex $dp 0]]]
    set sinks [list]
    foreach p [get_pins -quiet -of_objects $n -filter {DIRECTION==IN}] {
        lappend sinks [unit_of [get_cells -of_objects $p]]
    }
    set us [lsort -unique $sinks]
    if {[llength $us] == 0} { continue }
    if {[llength $us] == 1 && [lindex $us 0] eq $drv} { continue }
    puts $f "$n,$drv,[join $us {;}],[get_property FLAT_PIN_COUNT $n]"
}
close $f

# Device: capacity per clock region (sites, not cells).
set f [open $out_dir/clock_regions.csv w]
puts $f "cr,slr,x,y,slice,dsp,ramb36,uram"
foreach cr [get_clock_regions] {
    set s [get_sites -quiet -of_objects $cr]
    puts $f "$cr,[get_property NAME [get_slrs -of_objects $cr]],[get_property COLUMN_INDEX $cr],[get_property ROW_INDEX $cr],[llength [filter $s {SITE_TYPE =~ SLICE*}]],[llength [filter $s {SITE_TYPE =~ DSP*}]],[llength [filter $s {SITE_TYPE == RAMB36}]],[llength [filter $s {SITE_TYPE =~ URAM*}]]"
}
close $f
puts "EXTRACT_DONE"
exit
