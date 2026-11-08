#!/bin/bash
# run_hls.sh <cfg> : Vitis HLS csynth in D:\fp_exp\<cfg>
c=$1; cd /mnt/d/fp_exp/$c; s=$(date +%s)
cmd.exe /c "cd /d D:\\fp_exp\\$c && D:\\Vitis_HLS\\2023.2\\bin\\vitis_hls.bat -f hls.tcl -l hls.log" > hls_stdout.txt 2>&1
echo "HLS_DONE $c $(( $(date +%s)-s ))s rc=$? $(grep -cE '^ERROR' hls.log) errors"
