# Top-level connectivity of a synthesized checkpoint (units = top-level hier cells,
# top-level primitives = "top"). Writes nets.csv: driver unit, sink units, bit count per pair.
lassign $argv in_dcp out_dir
file mkdir $out_dir
open_checkpoint $in_dcp
set prim [dict create]
foreach c [get_cells -filter {IS_PRIMITIVE==1}] { dict set prim $c 1 }
proc u {c} { global prim; set h [lindex [split $c /] 0]; if {[dict exists $prim $h]} {return top}; return $h }
set pairs [dict create]
foreach n [get_nets -filter {TYPE != GROUND && TYPE != POWER}] {
    set pins [get_pins -quiet -of_objects $n]
    if {[llength $pins] < 2} continue
    set drv ""; set sinks {}
    foreach p $pins {
        set cu [u [get_property PARENT_CELL $p]]
        if {[get_property DIRECTION $p] eq "OUT"} { set drv $cu } else { lappend sinks $cu }
    }
    if {$drv eq ""} continue
    foreach s [lsort -unique $sinks] {
        if {$s eq $drv} continue
        dict incr pairs "$drv,$s"
    }
}
set f [open $out_dir/nets.csv w]
puts $f "driver,sinks,bits"
dict for {k v} $pairs { puts $f "$k,$v" }
close $f
puts NETS_DONE
exit
