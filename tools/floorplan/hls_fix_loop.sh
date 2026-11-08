#!/bin/bash
# Run HLS; on HLS 200-994 (dataflow-in-loop carried value) drop that loop's
# '#pragma HLS dataflow' and retry. Log each removal.
c=$1; d=/mnt/d/fp_exp/$c
for i in $(seq 1 40); do
  /mnt/d/fp_exp/run_hls.sh $c > /dev/null
  ln=$(grep -m1 "HLS 200-994" $d/hls.log | sed -E 's/.*design.cpp:([0-9]+).*/\1/')
  if [ -z "$ln" ]; then
    echo "$c attempt $i: $(grep -cE '^ERROR' $d/hls.log) errors; csynth rpt: $(ls $d/hls_prj/sol/syn/report/forward_csynth.rpt 2>/dev/null)"; exit 0
  fi
  nxt=$((ln+1))
  if sed -n "${nxt}p" $d/design.cpp | grep -q "pragma HLS dataflow"; then
    sed -i "${nxt}s|#pragma HLS dataflow|// dataflow removed (HLS 200-994)|" $d/design.cpp
    echo "$c attempt $i: removed dataflow at loop line $ln" >> $d/dataflow_removed.log
  else
    echo "$c: unexpected layout at $ln"; exit 1
  fi
done
