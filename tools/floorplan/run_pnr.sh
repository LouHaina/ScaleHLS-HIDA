#!/bin/bash
# run_pnr.sh <cfg> <fp: none|name (=> <cfg>/fp_<name>.tcl)> <directive> [stage place|route|resume] [strip full|x0-x1]
# resume: route the post_place.dcp of the matching place-stage run, in its out dir
cfg=$1; fp=$2; dir=$3; stage=${4:-route}; strip=${5:-full}
ost=$stage; [ "$stage" = resume ] && ost=place
out=$cfg/out_${fp}_${dir}_${ost}; cd /mnt/d/fp_exp; mkdir -p $out
f=none; [ "$fp" != none ] && f=$cfg/fp_$fp.tcl
cmd.exe /c "cd /d D:\\fp_exp && D:\\Vivado\\2023.2\\bin\\vivado.bat -mode batch -nojournal -log $out\\vivado_$stage.log -source run_pnr.tcl -tclargs $cfg $f $dir $out $stage $strip" > $out/stdout.txt 2>&1
echo "PNR $out $(cat $out/summary.txt 2>/dev/null || echo FAILED: $(grep -m2 -E '^ERROR' $out/vivado_$stage.log))"
