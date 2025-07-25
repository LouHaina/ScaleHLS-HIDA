add_mlir_library(MLIRScaleHLSTransforms
  Dataflow/BalanceDataflowNode.cpp
  Dataflow/BufferizeDataflow.cpp
  Dataflow/ConvertDataflowToFunc.cpp
  Dataflow/CreateDataflowFromAffine.cpp
  Dataflow/CreateDataflowFromLinalg.cpp
  Dataflow/CreateDataflowFromTosa.cpp
  Dataflow/CreateTokenStream.cpp
  Dataflow/EliminateMultiConsumer.cpp
  Dataflow/EliminateMultiProducer.cpp
  Dataflow/LegalizeDataflow.cpp
  Dataflow/LowerDataflow.cpp
  Dataflow/ParallelizeDataflowNode.cpp
  Dataflow/PlaceDataflowBuffer.cpp
  Dataflow/ScheduleDataflowNode.cpp
  Dataflow/StreamDataflowTask.cpp

  Directive/ArrayPartition.cpp
  Directive/CreateAxiInterface.cpp
  Directive/CreateHLSPrimitive.cpp
  Directive/FuncPipelining.cpp
  Directive/LoopPipelining.cpp
  Directive/LowerAffine.cpp
  Directive/QoREstimation.cpp

  Loop/AffineLoopFusion.cpp
  Loop/AffineLoopOrderOpt.cpp
  Loop/AffineLoopPerfection.cpp
  Loop/AffineLoopTile.cpp
  Loop/AffineLoopUnrollJam.cpp
  Loop/MaterializeReduction.cpp
  Loop/RemoveVariableBound.cpp

  Memory/AffineStoreForward.cpp
  Memory/BufferVectorize.cpp
  Memory/CollapseMemrefUnitDims.cpp
  Memory/CreateLocalBuffer.cpp
  Memory/CreateMemrefSubview.cpp
  Memory/LowerCopyToAffine.cpp
  Memory/RaiseAffineToCopy.cpp
  Memory/ReduceInitialInterval.cpp
  Memory/SimplifyAffineIf.cpp
  Memory/SimplifyCopy.cpp

  Tensor/ConvertTensorToLinalg.cpp
  Tensor/LinalgAnalyzeModel.cpp
  Tensor/LinalgFakeQuantize.cpp
  Tensor/TosaFakeQuantize.cpp
  Tensor/TosaSimplifyGraph.cpp

  DesignSpaceExplore.cpp
  FuncDuplication.cpp
  FuncPreprocess.cpp
  LayoutPartitionEnhanced.cpp
  Passes.cpp
  Utils.cpp

  DEPENDS
  MLIRScaleHLSTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRHLS
  )
