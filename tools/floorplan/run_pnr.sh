#!/bin/bash
# run_pnr.sh <cfg> <fp: none|name (=> <cfg>/fp_<name>.tcl)> <directive>
cfg=$1; fp=$2; dir=$3; out=$cfg/out_${fp}_${dir}; cd /mnt/d/fp_exp; mkdir -p $out
f=none; [ "$fp" != none ] && f=$cfg/fp_$fp.tcl
cmd.exe /c "cd /d D:\\fp_exp && D:\\Vivado\\2023.2\\bin\\vivado.bat -mode batch -nojournal -log $out\\vivado.log -source run_pnr.tcl -tclargs $cfg $f $dir $out" > $out/stdout.txt 2>&1
echo "PNR $out $(cat $out/summary.txt 2>/dev/null || echo FAILED: $(grep -m2 -E '^ERROR' $out/vivado.log))"
