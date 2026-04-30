# =============================================================================
# Vivado Physical Floorplan Constraints
# Generated from HIDA Layout Partitioning
# Target Device: xcvu37p-fsvh2892-2L-e (Alveo U250)
# Number of partitions: 4
# =============================================================================

# Partition Resource Requirements:
# Partition 0: LUT=0, FF=0, DSP=245, BRAM=2, URAM=0
# Partition 1: LUT=0, FF=0, DSP=184, BRAM=85, URAM=0
# Partition 2: LUT=100, FF=100, DSP=83, BRAM=98, URAM=0
# Partition 3: LUT=0, FF=0, DSP=229, BRAM=89, URAM=0

# Remove any existing pblocks
catch {delete_pblocks [get_pblocks]}

# =============================================================================
# Partition 0 - SLR0
# Resources: DSP=245, BRAM=2, LUT=0, FF=0
# =============================================================================
create_pblock pblock_partition_0
resize_pblock pblock_partition_0 -add {SLR0}
set_property CONTAIN_ROUTING true [get_pblocks pblock_partition_0]
set_property SNAPPING_MODE ON [get_pblocks pblock_partition_0]
resize_pblock pblock_partition_0 -add {DSP48E2_X0Y0:DSP48E2_X17Y47}
resize_pblock pblock_partition_0 -add {RAMB36_X0Y0:RAMB36_X7Y23 RAMB18_X0Y0:RAMB18_X15Y47}

# Add cells for partition 0 functions
set partition_0_cells [list]
catch {lappend partition_0_cells [get_cells -hierarchical -filter "REF_NAME == forward"]}
if {[llength $partition_0_cells] > 0} {
    add_cells_to_pblock pblock_partition_0 $partition_0_cells
    puts "Added [llength $partition_0_cells] cells to pblock_partition_0"
}

# =============================================================================
# Partition 1 - SLR1
# Resources: DSP=184, BRAM=85, LUT=0, FF=0
# =============================================================================
create_pblock pblock_partition_1
resize_pblock pblock_partition_1 -add {SLR1}
set_property CONTAIN_ROUTING true [get_pblocks pblock_partition_1]
set_property SNAPPING_MODE ON [get_pblocks pblock_partition_1]
resize_pblock pblock_partition_1 -add {DSP48E2_X0Y48:DSP48E2_X17Y95}
resize_pblock pblock_partition_1 -add {RAMB36_X0Y24:RAMB36_X7Y47 RAMB18_X0Y48:RAMB18_X15Y95}

# Add cells for partition 1 functions
set partition_1_cells [list]
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node82*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node83*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node84*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node85*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node86*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node87*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node88*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node89*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node130*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node131*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node132*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node133*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node134*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node135*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node136*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node137*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node138*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node139*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node15*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node16*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node17*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node18*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node19*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node92*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node93*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node94*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node95*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node96*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node91*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node98*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node99*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node20*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node22*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node23*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node25*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node26*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node27*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node28*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node29*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node30*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node31*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node32*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node33*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node34*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node35*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node36*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node38*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node162*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node163*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node164*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node165*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node166*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node167*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node168*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node169*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node40*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node41*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node42*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node43*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node45*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node46*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node48*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node49*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node3*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node4*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node5*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node171*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node170*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node50*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node51*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node52*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node53*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node54*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node55*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node56*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node57*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node58*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node101*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node102*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node103*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node104*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node105*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node107*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node109*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node68*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node69*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node110*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node111*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node112*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node70*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node71*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node72*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node73*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node74*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node125*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node126*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node127*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node128*"]}
catch {lappend partition_1_cells [get_cells -hierarchical -filter "NAME =~ *forward_node129*"]}
if {[llength $partition_1_cells] > 0} {
    add_cells_to_pblock pblock_partition_1 $partition_1_cells
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
resize_pblock pblock_partition_2 -add {SLICE_X0Y240:SLICE_X107Y359}
resize_pblock pblock_partition_2 -add {DSP48E2_X0Y96:DSP48E2_X17Y143}
resize_pblock pblock_partition_2 -add {RAMB36_X0Y48:RAMB36_X7Y71 RAMB18_X0Y96:RAMB18_X15Y143}

# Add cells for partition 2 functions
set partition_2_cells [list]
catch {lappend partition_2_cells [get_cells -hierarchical -filter "NAME =~ *forward_node14*"]}
catch {lappend partition_2_cells [get_cells -hierarchical -filter "NAME =~ *forward_node90*"]}
catch {lappend partition_2_cells [get_cells -hierarchical -filter "NAME =~ *forward_node97*"]}
catch {lappend partition_2_cells [get_cells -hierarchical -filter "NAME =~ *forward_node21*"]}
catch {lappend partition_2_cells [get_cells -hierarchical -filter "NAME =~ *forward_node37*"]}
catch {lappend partition_2_cells [get_cells -hierarchical -filter "NAME =~ *forward_node161*"]}
catch {lappend partition_2_cells [get_cells -hierarchical -filter "NAME =~ *forward_node44*"]}
catch {lappend partition_2_cells [get_cells -hierarchical -filter "NAME =~ *forward_node106*"]}
catch {lappend partition_2_cells [get_cells -hierarchical -filter "NAME =~ *main*"]}
if {[llength $partition_2_cells] > 0} {
    add_cells_to_pblock pblock_partition_2 $partition_2_cells
    puts "Added [llength $partition_2_cells] cells to pblock_partition_2"
}

# =============================================================================
# Partition 3 - SLR3
# Resources: DSP=229, BRAM=89, LUT=0, FF=0
# =============================================================================
create_pblock pblock_partition_3
resize_pblock pblock_partition_3 -add {SLR3}
set_property CONTAIN_ROUTING true [get_pblocks pblock_partition_3]
set_property SNAPPING_MODE ON [get_pblocks pblock_partition_3]
resize_pblock pblock_partition_3 -add {DSP48E2_X0Y144:DSP48E2_X17Y191}
resize_pblock pblock_partition_3 -add {RAMB36_X0Y72:RAMB36_X7Y95 RAMB18_X0Y144:RAMB18_X15Y191}

# Add cells for partition 3 functions
set partition_3_cells [list]
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node80*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node81*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node10*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node11*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node12*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node13*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node140*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node141*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node142*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node143*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node144*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node145*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node146*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node147*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node148*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node149*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node24*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node150*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node151*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node152*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node153*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node154*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node155*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node156*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node157*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node158*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node159*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node39*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node160*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node47*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node0*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node1*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node2*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node6*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node7*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node8*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node9*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node59*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node100*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node108*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node60*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node61*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node62*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node63*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node64*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node65*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node66*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node67*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node113*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node114*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node115*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node116*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node117*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node118*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node119*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node75*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node76*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node77*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node78*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node79*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node120*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node121*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node122*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node123*"]}
catch {lappend partition_3_cells [get_cells -hierarchical -filter "NAME =~ *forward_node124*"]}
if {[llength $partition_3_cells] > 0} {
    add_cells_to_pblock pblock_partition_3 $partition_3_cells
    puts "Added [llength $partition_3_cells] cells to pblock_partition_3"
}

# =============================================================================
# Inter-SLR Routing Optimization
# =============================================================================
set_property ROUTE.SLR_CROSSING_MINIMIZATION true [current_design]
set_property ROUTE.SLR_CROSSING_OPTIMIZATION true [current_design]
set_property MAX_FANOUT 50 [get_nets -hierarchical]

puts "=== Floorplan constraints applied successfully ==="
