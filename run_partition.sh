./build/bin/scalehls-opt ./samples/pytorch/resnet18/resnet18.mlir \
  --hida-pytorch-pipeline="top-func=forward loop-tile-size=4 loop-unroll-factor=2 enable-layout-partition=true num-tiles=3 partition-tcl-output=partition_0.3_1.tcl partition-balance-threshold=0.3 partition-enable-comm-opt=true partition-single-slr-util-threshold=0.3" \
  > output.mlir
