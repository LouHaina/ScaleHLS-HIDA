# =============================================================================
# Vivado Physical Floorplan Constraints
# Generated from HIDA Layout Partitioning
# Target Device: xcvu37p-fsvh2892-2L-e (Alveo U250)
# Number of partitions: 4
# =============================================================================

proc get_func_cells {funcName} {

    set patt "*${funcName}*_U0"
    set c3 [get_cells -hier -quiet -filter "NAME =~ $patt"]
    if {[llength $c3] == 0} {
        return {}
    }

    return [lsort -unique $c3]
}
# Partition Resource Requirements:
# Partition 0: LUT=0, FF=0, DSP=245, BRAM=2, URAM=0
# Partition 1: LUT=0, FF=0, DSP=184, BRAM=85, URAM=0
# Partition 2: LUT=100, FF=100, DSP=83, BRAM=98, URAM=0
# Partition 3: LUT=0, FF=0, DSP=229, BRAM=89, URAM=0

# Remove any existing pblocks
catch {delete_pblocks [get_pblocks]}

set_param tcl.collectionResultDisplayLimit 0
# =============================================================================
# Partition 1 - SLR1
# Resources: DSP=184, BRAM=85, LUT=0, FF=0
# =============================================================================
create_pblock pblock_partition_1
resize_pblock pblock_partition_1 -add {SLR1}
set_property CONTAIN_ROUTING true [get_pblocks pblock_partition_1]
set_property SNAPPING_MODE ON [get_pblocks pblock_partition_1]


# Add cells for partition 1 functions
set partition_1_cells [list]
set _cells [get_func_cells forward_node82]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node85]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node133]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node135]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node16]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node92]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node30]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node32]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node166]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node168]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node3]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node4]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node52]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node54]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node68]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node70]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node125]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }
set _cells [get_func_cells forward_node128]
if {[llength $_cells] > 0} { lappend partition_1_cells {*}${_cells} }

if {[llength $partition_1_cells] > 0} {
    puts "Partition 1: collected [llength $partition_1_cells] cells"
    add_cells_to_pblock pblock_partition_1 {*}$partition_1_cells
    puts "Added [llength $partition_1_cells] cells to pblock_partition_1"
}

# =============================================================================
# Partition 2 - SLR2
# Resources: DSP=83, BRAM=98, LUT=100, FF=100
# =============================================================================
create_pblock pblock_partition_2
resize_pblock pblock_partition_2 -add {SLR2}
set_property CONTAIN_ROUTING true [get_pblocks pblock_partition_2]
set_property SNAPPING_MODE ON [get_pblocks pblock_partition_2]


# Add cells for partition 2 functions
set partition_2_cells [list]
set _cells [get_func_cells forward_node14]
if {[llength $_cells] > 0} { lappend partition_2_cells {*}${_cells} }
set _cells [get_func_cells forward_node90]
if {[llength $_cells] > 0} { lappend partition_2_cells {*}${_cells} }
set _cells [get_func_cells forward_node97]
if {[llength $_cells] > 0} { lappend partition_2_cells {*}${_cells} }
set _cells [get_func_cells forward_node21]
if {[llength $_cells] > 0} { lappend partition_2_cells {*}${_cells} }
set _cells [get_func_cells forward_node37]
if {[llength $_cells] > 0} { lappend partition_2_cells {*}${_cells} }
set _cells [get_func_cells forward_node161]
if {[llength $_cells] > 0} { lappend partition_2_cells {*}${_cells} }
set _cells [get_func_cells forward_node44]
if {[llength $_cells] > 0} { lappend partition_2_cells {*}${_cells} }
set _cells [get_func_cells forward_node106]
if {[llength $_cells] > 0} { lappend partition_2_cells {*}${_cells} }
set _cells [get_func_cells main]
if {[llength $_cells] > 0} { lappend partition_2_cells {*}${_cells} }

if {[llength $partition_2_cells] > 0} {
    puts "Partition 2: collected [llength $partition_1_cells] cells"
    add_cells_to_pblock pblock_partition_2 {*}$partition_2_cells
    puts "Added [llength $partition_2_cells] cells to pblock_partition_2"
}

# =============================================================================
# Partition 3 - SLR3
# Resources: DSP=229, BRAM=89, LUT=0, FF=0
# =============================================================================
create_pblock pblock_partition_3
resize_pblock pblock_partition_3 -add {SLR0}
set_property CONTAIN_ROUTING true [get_pblocks pblock_partition_3]
set_property SNAPPING_MODE ON [get_pblocks pblock_partition_3]


# Add cells for partition 3 functions
set partition_3_cells [list]
set _cells [get_func_cells forward_node140]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node143]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node149]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node24]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node151]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node156]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node39]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node47]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node0]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node1]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node6]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node9]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node59]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node100]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node108]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node62]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node113]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node114]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node119]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node75]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
set _cells [get_func_cells forward_node77]
if {[llength $_cells] > 0} { lappend partition_3_cells {*}${_cells} }
if {[llength $partition_3_cells] > 0} {
    puts "Partition 3: collected [llength $partition_1_cells] cells"
    add_cells_to_pblock pblock_partition_3 {*}$partition_3_cells
    puts "Added [llength $partition_3_cells] cells to pblock_partition_3"
}



puts "=== Floorplan constraints applied successfully ==="
