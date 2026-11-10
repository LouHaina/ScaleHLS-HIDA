//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "scalehls/Dialect/HLS/Analysis.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "parallelize-dataflow-node"

using namespace mlir;
using namespace scalehls;

/// Apply loop vectorization to the loop band.
static bool applyLoopVectorization(AffineLoopBand &band,
                                   FactorList vectorFactors) {
  assert(!band.empty() && "no loops provided");
  if (llvm::all_of(vectorFactors, [](unsigned factor) { return factor == 1; }))
    return true;

  // We require all loops to be parallel loop.
  for (auto [loop, size] : llvm::zip(band, vectorFactors))
    if (size != 1 && !(hasParallelAttr(loop) || isLoopParallel(loop)))
      return false;

  LLVM_DEBUG(llvm::dbgs() << "Loop vectorization ";);
  LLVM_DEBUG(llvm::dbgs() << "vector factors: ";);
  LLVM_DEBUG(for (auto factor : vectorFactors) llvm::dbgs() << factor << " ";);

  // Apply loop vectorization.
  auto loopSet = llvm::DenseSet<Operation *>(band.begin(), band.end());
  auto sizes =
      SmallVector<int64_t>(vectorFactors.rbegin(), vectorFactors.rend());
  vectorizeAffineLoops(band.front()->getParentOp(), loopSet, sizes, {});
  return true;
}

namespace {
struct GenerateBufferLayout
    : public OpInterfaceRewritePattern<VectorTransferOpInterface> {
  using OpInterfaceRewritePattern<
      VectorTransferOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(VectorTransferOpInterface transferOp,
                                PatternRewriter &rewriter) const override {
    auto vectorType = transferOp.getVectorType();
    // TODO: For now, we demand the vector rank to be the same as memref rank.
    if (vectorType.getRank() != transferOp.getShapedType().getRank())
      return failure();

    // The default tile shape is as same as the memref shape.
    auto tileShape = transferOp.getShapedType().getShape();

    // If the source has a tile layout, use it as the tile shape.
    if (auto layout = getTileLayout(transferOp.source())) {
      // If the layout is already vectorized, check if it's vector shape is
      // compatible with the existing one. If not, it means that different
      // transfer ops are using different vector shapes, which is not allowed.
      if (layout.isVectorized()) {
        if (vectorType.getShape() != layout.getVectorShape())
          return transferOp->emitOpError("incompatible vector shape");
        else
          return failure();
      }
      tileShape = layout.getTileShape();
    }

    // Set the tile layout the calculated tile shape and vector shape.
    setTileLayout(transferOp.source(), tileShape, vectorType.getShape());
    return success();
  }
};
} // namespace

namespace {
struct ParallelizeDataflowNode
    : public ParallelizeDataflowNodeBase<ParallelizeDataflowNode> {
  ParallelizeDataflowNode() = default;
  ParallelizeDataflowNode(unsigned loopUnrollFactor, bool unrollPointLoopOnly,
                          bool argComplexityAware, bool argCorrelationAware,
                          std::string argUnitFactors = "",
                          bool argRecordUnitFactors = false) {
    maxUnrollFactor = loopUnrollFactor;
    pointLoopOnly = unrollPointLoopOnly;
    complexityAware = argComplexityAware;
    correlationAware = argCorrelationAware;
    unitFactors = argUnitFactors;
    recordUnitFactors = argRecordUnitFactors;
  }

  /// Read the per-unit unroll factors chosen by the SLR budget solver
  /// (tools/floorplan/floorplan.py budget). A unit is a node of the top-level
  /// schedule; the list is indexed by the program order of these nodes.
  bool loadUnitFactors(func::FuncOp func) {
    requestedUnitFactors.clear();
    if (unitFactors.empty())
      return true;
    auto file = llvm::MemoryBuffer::getFile(unitFactors);
    if (!file) {
      func.emitError("cannot read unit factors file: ") << unitFactors;
      return false;
    }
    auto parsed = llvm::json::parse(file.get()->getBuffer());
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      func.emitError("cannot parse unit factors json: ") << unitFactors;
      return false;
    }
    auto *root = parsed->getAsObject();
    auto *list = root ? root->getArray("unit_factors") : nullptr;
    if (!list) {
      func.emitError("unit factors json has no \"unit_factors\" array");
      return false;
    }
    for (auto &value : *list) {
      auto factor = value.getAsInteger();
      requestedUnitFactors.push_back(factor && *factor > 0 ? *factor : 0);
    }
    return true;
  }

  /// Try to calculate the unroll factors of the nodes contained in each
  /// dataflow schedule.
  void getNodeParallelFactorMap(func::FuncOp func) {
    auto compAnal = ComplexityAnalysis(func);
    nodeParallelFactorMap.clear();
    appliedUnitFactors.clear();

    func.walk<WalkOrder::PreOrder>([&](ScheduleOp schedule) {
      unsigned long scheduleUnrollFactor = maxUnrollFactor.getValue();
      if (auto parentNode = schedule->getParentOfType<NodeOp>()) {
        if (!nodeParallelFactorMap.count(parentNode)) {
          parentNode.emitOpError("failed to get parent node's unroll factor");
          return WalkResult::interrupt();
        }
        scheduleUnrollFactor = nodeParallelFactorMap.lookup(parentNode);
        // FIXME: A hacky method to hand tune the factors and resolve
        // outstanding dataflow nodes.
        if (auto attr = schedule->getAttr("increase"))
          if (auto annoFactor = attr.dyn_cast<IntegerAttr>())
            scheduleUnrollFactor *= annoFactor.getInt();
        if (auto attr = schedule->getAttr("decrease"))
          if (auto annoFactor = attr.dyn_cast<IntegerAttr>())
            scheduleUnrollFactor /= annoFactor.getInt();
      }

      auto scheduleComplexity = compAnal.getScheduleComplexity(schedule);
      if (!scheduleComplexity.has_value()) {
        schedule.emitOpError("failed to get schedule complexity");
        return WalkResult::interrupt();
      }

      for (auto node : schedule.getOps<NodeOp>()) {
        auto nodeComplexity = compAnal.getNodeComplexity(node);
        if (!nodeComplexity.has_value()) {
          node.emitOpError("failed to get node complexity");
          return WalkResult::interrupt();
        }
        auto nodeUnrollFactor = 1 + scheduleUnrollFactor *
                                        nodeComplexity.value() /
                                        scheduleComplexity.value();

        // Nodes of a top-level schedule are the floorplan units. Their factor
        // can be overridden by the SLR budget, and is recorded so that the
        // exported graph tells which factor produced which area.
        if (!schedule->getParentOfType<NodeOp>()) {
          auto unitIndex = appliedUnitFactors.size();
          if (unitIndex < requestedUnitFactors.size() &&
              requestedUnitFactors[unitIndex] > 0)
            nodeUnrollFactor = requestedUnitFactors[unitIndex];
          appliedUnitFactors.push_back(nodeUnrollFactor);
        }
        nodeParallelFactorMap.insert({node, nodeUnrollFactor});

        LLVM_DEBUG(
            // clang-format off
          llvm::dbgs() << "\nNode Complexity: " << nodeComplexity.value() << "\n";
          llvm::dbgs() << "Schedule Complexity: " << scheduleComplexity.value() << "\n";
          llvm::dbgs() << "Node Factor: " << nodeUnrollFactor << "\n";
          llvm::dbgs() << "Schedule Factor: " << scheduleUnrollFactor << "\n";
          llvm::dbgs() << "Node at " << node.getLoc() << ": \n" << node << "\n";
            // clang-format on
        );
      }
      return WalkResult::advance();
    });
  }

  /// Unroll dataflow node with the given parallel factor. If the pass is not
  /// complexity aware, always unroll with the max unroll factor.
  void applyNaiveLoopUnroll(NodeOp node, unsigned parallelFactor) {
    auto unrollFactor = parallelFactor;
    if (!complexityAware)
      unrollFactor = maxUnrollFactor.getValue();

    // Collect all loop bands to be unrolled.
    AffineLoopBands bands;
    node.walk([&](AffineForOp loop) {
      if (loop->getParentOfType<NodeOp>() == node &&
          loop.getOps<mlir::AffineForOp>().empty() &&
          loop.getOps<ScheduleOp>().empty()) {
        AffineLoopBand band;
        getLoopBandFromInnermost(loop, band);
        bands.push_back(band);
      }
    });

    for (auto &band : bands) {
      // For loop band that has effect on external buffers, we should directly
      // unroll them without considering whether it's point loop.
      // FIXME: Need a better solution on handling external buffers.
      if (pointLoopOnly && !hasEffectOnExternalBuffer(band.front())) {
        AffineLoopBand tileBand;
        AffineLoopBand pointBand;
        if (!getTileAndPointLoopBand(band, tileBand, pointBand) ||
            pointBand.empty())
          continue;
        band = pointBand;
      }

      auto factors = FactorList(band.size(), 1);
      SmallVector<FactorList> constrFactors;
      if (failed(getEvenlyDistributedFactors(unrollFactor, factors, band,
                                             constrFactors,
                                             /*powerOf2Constr=*/true)))
        factors = getDistributedFactors(unrollFactor, band);
      applyLoopUnrollJam(band, factors);
    }
  }

  /// Unroll loops based on the correlations between dataflow nodes.
  void applyCorrelationAwareUnroll(func::FuncOp func) {
    auto corrAnal = CorrelationAnalysis(func);

    // We first sort all nodes in a decending order of their associated number
    // of correlations. The rationale is nodes that have more correlations
    // should be optimized first.
    SmallVector<std::pair<NodeOp, unsigned>> worklist;
    for (auto &nodeAndList : corrAnal)
      worklist.push_back({nodeAndList.first, nodeAndList.second.size()});
    llvm::sort(worklist, [](auto a, auto b) { return a.second < b.second; });

    // Optimize the unroll factors from the most critical node.
    llvm::SmallDenseMap<NodeOp, FactorList> nodeUnrollFactorsMap;
    llvm::SmallDenseMap<NodeOp, unsigned> nodeVisitCount;
    while (!worklist.empty()) {
      auto current = worklist.pop_back_val();
      auto node = current.first;
      auto corrNum = current.second;

      // Nodes sharing an external buffer must end up with identical factors,
      // so a node is revisited whenever its neighbour got different ones. If
      // the two sides have different parallel factors (which per-unit factors
      // make common), redistributing each side in turn can alternate forever.
      // After a few visits a node therefore adopts the factors required by
      // its external-buffer neighbours instead of redistributing its own.
      const unsigned maxRedistributions = 8;
      bool adoptNeighbourFactors = ++nodeVisitCount[node] > maxRedistributions;

      // If the correlation list is empty, which means the correlation analysis
      // failed, skip the current node.
      auto corrList = corrAnal.getCorrelations(node);
      if (corrList.empty())
        continue;

      // Get the parallel factor and loop band associated with the current node.
      // Also initialize the unroll factors as one.
      auto parallelFactor = maxUnrollFactor.getValue();
      if (complexityAware && nodeParallelFactorMap.count(node))
        parallelFactor = nodeParallelFactorMap.lookup(node);
      auto band = getNodeLoopBand(node);
      auto factors = FactorList(band.size(), 1);

      // Collect the loop unroll factors of all correlated nodes that have been
      // traversed. Additionally, collect the node and factors of external
      // buffer correlations for later use.
      SmallVector<FactorList> corrFactorsList;
      SmallVector<std::pair<NodeOp, FactorList>> candidatesToRevisit;
      for (auto corr : corrList) {
        auto corrNode = corr.getCorrelatedNode(node);
        if (!nodeUnrollFactorsMap.count(corrNode))
          continue;

        // Permute the unroll factors of correlated node with the permutation
        // map recorded in the correlation.
        auto [corrFactors, roundedFlag] = corr.permuteAndScaleFactors(
            corrNode, nodeUnrollFactorsMap.lookup(corrNode));
        corrFactorsList.push_back(corrFactors);

        // Rounded parallel factors means that when the correlated node is
        // parallelized, the strided memory access pattern of the current node
        // was not awared. Therefore, we need to revisit the correlated node.
        // if (roundedFlag)
        //   worklist.push_back({corrNode, 0});

        // Make sure each factor is larger than the corresponding factor of
        // the external buffer correlatations.
        if (isExtBuffer(corr.getBuffer().getMemref())) {
          FactorList newFactors;
          for (auto [factor, corrFactor] : llvm::zip(factors, corrFactors))
            newFactors.push_back(std::max(factor, corrFactor));
          factors = newFactors;
          candidatesToRevisit.push_back({corrNode, corrFactors});
        }

        LLVM_DEBUG(
            // clang-format off
            llvm::dbgs() << "----------\n";
            llvm::dbgs() << "Shared buffer at " << corr.getBuffer().getLoc()
                         << ": " << *corr.getBuffer() << "\n";
            llvm::dbgs() << "Correlate Map: { ";
            for (auto factor : corr.getCorrelateMap(node))
              llvm::dbgs() << factor << " ";
            llvm::dbgs() << "}\n";
            llvm::dbgs() << "Correlated Factors: { ";
            for (auto factor : corrFactors)
              llvm::dbgs() << factor << " ";
            llvm::dbgs() << "}\n";
            llvm::dbgs() << "Correlated Node at " << corrNode.getLoc() << ": \n"
                         << corrNode << "\n";
            // clang-format on
        );
      }

      // Distribute factor based on the constraints and record in the map.
      // When adopting, "factors" already holds the element-wise maximum of
      // the external-buffer neighbours' factors and is kept as is.
      if (adoptNeighbourFactors && !candidatesToRevisit.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "Adopt neighbour factors\n");
      } else if (failed(getEvenlyDistributedFactors(parallelFactor, factors,
                                                    band, corrFactorsList,
                                                    /*powerOf2Constr=*/true)))
        factors = getDistributedFactors(parallelFactor, band);
      nodeUnrollFactorsMap[node] = factors;

      // If the final factors are not equal to the factors of any external
      // buffer correlated node, we need to recalculate the correlated node as
      // the factors must be identical eventually. Otherwise, we will not be
      // able to reason the suitable external buffer layout.
      for (auto [corrNode, corrFactors] : candidatesToRevisit)
        if (factors != corrFactors)
          worklist.push_back({corrNode, 0});

      LLVM_DEBUG(
          // clang-format off
          llvm::dbgs() << "\nCorrelations: " << corrNum << "\n";
          llvm::dbgs() << "Parallel: " << parallelFactor << "\n";
          llvm::dbgs() << "Factors: { ";
          for (auto factor : factors)
            llvm::dbgs() << factor << " ";
          llvm::dbgs() << "}\n";
          llvm::dbgs() << "Node at " << node.getLoc() << ": \n" << node << "\n";
          llvm::dbgs() << "==========\n";
          // clang-format on
      );
    }

    // Apply unroll and jam to loops that is successfully calculated for
    // correlation-aware unroll factors.
    for (auto p : nodeUnrollFactorsMap) {
      auto band = getNodeLoopBand(p.first);
      // if (hasEffectOnExternalBuffer(band.front()))
      //   applyLoopVectorization(band, p.second);
      // else
      applyLoopUnrollJam(band, p.second);
    }

    // Apply naive unroll to other loops.
    for (auto p : nodeParallelFactorMap)
      if (!nodeUnrollFactorsMap.count(p.first))
        applyNaiveLoopUnroll(p.first, p.second);
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();
    if (!loadUnitFactors(func))
      return signalPassFailure();
    getNodeParallelFactorMap(func);

    // A stale factor file (different number of units) must not be applied to
    // the wrong nodes silently.
    if (!appliedUnitFactors.empty() && !requestedUnitFactors.empty() &&
        appliedUnitFactors.size() != requestedUnitFactors.size()) {
      func.emitError("unit factors file lists ")
          << requestedUnitFactors.size() << " units but the function has "
          << appliedUnitFactors.size() << " top-level dataflow nodes";
      return signalPassFailure();
    }
    if (!appliedUnitFactors.empty() &&
        (recordUnitFactors || !unitFactors.empty()))
      func->setAttr("unit_unroll_factors",
                    Builder(context).getI64ArrayAttr(appliedUnitFactors));

    if (correlationAware)
      applyCorrelationAwareUnroll(func);
    else {
      for (auto p : nodeParallelFactorMap)
        applyNaiveLoopUnroll(p.first, p.second);
    }

    mlir::RewritePatternSet patterns(context);
    patterns.add<GenerateBufferLayout>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }

private:
  llvm::SmallDenseMap<NodeOp, unsigned long> nodeParallelFactorMap;
  SmallVector<int64_t> requestedUnitFactors;
  SmallVector<int64_t> appliedUnitFactors;
};
} // namespace

std::unique_ptr<Pass> scalehls::createParallelizeDataflowNodePass(
    unsigned loopUnrollFactor, bool unrollPointLoopOnly, bool complexityAware,
    bool correlationAware, std::string unitFactors, bool recordUnitFactors) {
  return std::make_unique<ParallelizeDataflowNode>(
      loopUnrollFactor, unrollPointLoopOnly, complexityAware, correlationAware,
      unitFactors, recordUnitFactors);
}
