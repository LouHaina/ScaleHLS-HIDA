//===- LayoutPartitionEnhanced.cpp - Enhanced Layout Partitioning -*- C++ -*-===//
//
// Enhanced layout-driven partitioning optimization pass for HIDA.
// Implements resource-aware partitioning with communication minimization.
//
//===----------------------------------------------------------------------===//

#include "scalehls/Transforms/Passes.h"
#include "scalehls/Dialect/HLS/HLS.h"
#include "scalehls/Dialect/HLS/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#define DEBUG_TYPE "scalehls-layout-partition"

using namespace mlir;
using namespace scalehls;
using namespace hls;

/// Hardware resource information for a function or partition
struct HardwareResource {
  uint64_t lut = 0;
  uint64_t ff = 0;
  uint64_t dsp = 0;
  uint64_t bram = 0;
  uint64_t uram = 0;

  HardwareResource() = default;

  HardwareResource(uint64_t l, uint64_t f, uint64_t d, uint64_t b, uint64_t u = 0)
      : lut(l), ff(f), dsp(d), bram(b), uram(u) {}

  HardwareResource &operator+=(const HardwareResource &other) {
    lut += other.lut;
    ff += other.ff;
    dsp += other.dsp;
    bram += other.bram;
    uram += other.uram;
    return *this;
  }

  HardwareResource operator+(const HardwareResource &other) const {
    HardwareResource result = *this;
    result += other;
    return result;
  }

  /// Check if this resource fits within the given constraints
  bool fitsWithin(const HardwareResource &constraint) const {
    return lut <= constraint.lut && ff <= constraint.ff &&
           dsp <= constraint.dsp && bram <= constraint.bram &&
           uram <= constraint.uram;
  }

  /// Calculate resource utilization ratio (0.0 to 1.0+)
  double getUtilizationRatio(const HardwareResource &constraint) const {
    double maxRatio = 0.0;
    if (constraint.lut > 0) maxRatio = std::max(maxRatio, (double)lut / constraint.lut);
    if (constraint.ff > 0) maxRatio = std::max(maxRatio, (double)ff / constraint.ff);
    if (constraint.dsp > 0) maxRatio = std::max(maxRatio, (double)dsp / constraint.dsp);
    if (constraint.bram > 0) maxRatio = std::max(maxRatio, (double)bram / constraint.bram);
    if (constraint.uram > 0) maxRatio = std::max(maxRatio, (double)uram / constraint.uram);
    return maxRatio;
  }

  /// Get total resource "weight" for sorting
  uint64_t getTotalWeight() const {
    // Weight DSP and BRAM higher as they are typically more constrained
    return lut + ff + (dsp * 10) + (bram * 8) + (uram * 12);
  }
};

/// Function information for partitioning
struct FunctionInfo {
  HardwareResource resource;
  SmallVector<StringRef, 4> callees;  // Functions this function calls
  SmallVector<StringRef, 4> callers;  // Functions that call this function
  int partitionId = -1;
  uint64_t communicationWeight = 0;  // Total communication cost
};

/// Enhanced Layout Partition Pass
struct LayoutPartitionPass : public PassWrapper<LayoutPartitionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutPartitionPass)

  LayoutPartitionPass() = default;

  // Pass parameters (will be set via the factory function)
  int numTiles = 2;
  std::string tclOutputPath = "partition.tcl";
  double balanceThreshold = 0.3;
  bool enableCommunicationOpt = true;

  /// Return the command line argument used when registering this pass
  StringRef getArgument() const override {
    return "scalehls-layout-partition";
  }

  /// Return the command line description used when registering this pass
  StringRef getDescription() const override {
    return "Enhanced layout-driven partitioning optimization for HIDA";
  }

  void runOnOperation() override {
    if (numTiles <= 0) numTiles = 1;

    ModuleOp module = getOperation();

    // Step 1: Collect function information and resources
    if (failed(collectFunctionInfo(module))) {
      signalPassFailure();
      return;
    }

    // Step 2: Build communication graph
    buildCommunicationGraph();

    // Step 3: Run resource-aware partitioning algorithm
    if (failed(runResourceAwarePartitioning())) {
      signalPassFailure();
      return;
    }

    // Step 4: Annotate functions with partition IDs
    annotateFunctions(module);

    // Step 5: Generate TCL script for Vivado floorplanning
    generateTclScript(module);

    // Debug output
    printPartitioningResults();
  }

private:
  llvm::StringMap<FunctionInfo> functionInfos;
  SmallVector<HardwareResource> partitionResources;
  HardwareResource totalResources;

  /// Step 1: Collect function information from QoR estimation
  LogicalResult collectFunctionInfo(ModuleOp module) {
    llvm::errs() << "=== LAYOUT PARTITION DEBUG: Collecting function information...\n";

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isDeclaration()) continue;

      auto &info = functionInfos[func.getName()];

      // Get resource information from QoR estimation pass
      if (auto resourceAttr = getResource(func)) {
        info.resource.lut = resourceAttr.getLut();
        info.resource.dsp = resourceAttr.getDsp();
        info.resource.bram = resourceAttr.getBram();
        // Note: FF and URAM not directly available in current ResourceAttr
        info.resource.ff = info.resource.lut; // Approximate FF as LUT count
        info.resource.uram = 0; // Will be enhanced later

        totalResources += info.resource;

        llvm::errs() << "=== LAYOUT PARTITION DEBUG: Function " << func.getName()
                     << ": LUT=" << info.resource.lut
                     << ", DSP=" << info.resource.dsp
                     << ", BRAM=" << info.resource.bram << "\n";
      } else {
        // Fallback: estimate resources based on function complexity
        info.resource.lut = 100;  // Default estimate
        info.resource.ff = 100;
        info.resource.dsp = 1;
        info.resource.bram = 1;

        llvm::errs() << "=== LAYOUT PARTITION DEBUG: Function " << func.getName()
                     << ": Using default resource estimates\n";
      }

      // Collect function call relationships
      llvm::SetVector<StringRef> callees;
      func.walk([&](func::CallOp call) {
        callees.insert(call.getCallee());
      });
      info.callees.append(callees.begin(), callees.end());
    }

    llvm::errs() << "=== LAYOUT PARTITION DEBUG: Total resources: LUT=" << totalResources.lut
                 << ", DSP=" << totalResources.dsp
                 << ", BRAM=" << totalResources.bram << "\n";

    return success();
  }

  /// Step 2: Build bidirectional communication graph
  void buildCommunicationGraph() {
    llvm::errs() << "=== LAYOUT PARTITION DEBUG: Building communication graph...\n";

    // Build caller-callee relationships
    for (auto &entry : functionInfos) {
      StringRef funcName = entry.first();
      auto &info = entry.second;
      for (auto callee : info.callees) {
        if (functionInfos.count(callee)) {
          functionInfos[callee].callers.push_back(funcName);
        }
      }
    }

    // Calculate communication weights (simplified model)
    for (auto &entry : functionInfos) {
      auto &info = entry.second;
      info.communicationWeight = info.callees.size() + info.callers.size();
    }
  }

  /// Step 3: Resource-aware partitioning algorithm
  LogicalResult runResourceAwarePartitioning() {
    llvm::errs() << "=== LAYOUT PARTITION DEBUG: Running resource-aware partitioning...\n";

    // Initialize partition resources
    partitionResources.assign(numTiles, HardwareResource());

    // Calculate target resources per partition
    HardwareResource targetPerPartition;
    targetPerPartition.lut = (totalResources.lut + numTiles - 1) / numTiles;
    targetPerPartition.ff = (totalResources.ff + numTiles - 1) / numTiles;
    targetPerPartition.dsp = (totalResources.dsp + numTiles - 1) / numTiles;
    targetPerPartition.bram = (totalResources.bram + numTiles - 1) / numTiles;
    targetPerPartition.uram = (totalResources.uram + numTiles - 1) / numTiles;

    // Sort functions by resource weight (largest first)
    SmallVector<StringRef> sortedFunctions;
    for (auto &entry : functionInfos) {
      StringRef funcName = entry.first();
      sortedFunctions.push_back(funcName);
    }

    llvm::sort(sortedFunctions, [&](StringRef a, StringRef b) {
      return functionInfos[a].resource.getTotalWeight() >
             functionInfos[b].resource.getTotalWeight();
    });

    // Assign functions to partitions using enhanced greedy algorithm
    for (StringRef funcName : sortedFunctions) {
      auto &funcInfo = functionInfos[funcName];
      int bestPartition = findBestPartition(funcName, targetPerPartition);

      if (bestPartition == -1) {
        llvm::errs() << "=== LAYOUT PARTITION DEBUG: Failed to find suitable partition for "
                     << funcName << "\n";
        return failure();
      }

      funcInfo.partitionId = bestPartition;
      partitionResources[bestPartition] += funcInfo.resource;

      llvm::errs() << "Assigned " << funcName
                   << " to partition " << bestPartition << "\n";
    }

    return success();
  }

  /// Find the best partition for a function considering resources and communication
  int findBestPartition(StringRef funcName, const HardwareResource &targetPerPartition) {
    auto &funcInfo = functionInfos[funcName];
    int bestPartition = -1;
    double bestScore = std::numeric_limits<double>::max();
    int leastFullPartition = -1;
    double minUtilization = std::numeric_limits<double>::max();

    for (int partitionId = 0; partitionId < numTiles; ++partitionId) {
      auto &partitionRes = partitionResources[partitionId];
      auto newPartitionRes = partitionRes + funcInfo.resource;

      // Track the least full partition as fallback
      double currentUtilization = partitionRes.getUtilizationRatio(targetPerPartition);
      if (currentUtilization < minUtilization) {
        minUtilization = currentUtilization;
        leastFullPartition = partitionId;
      }

      // Check if resources fit (with more flexible tolerance for large functions)
      double utilizationRatio = newPartitionRes.getUtilizationRatio(targetPerPartition);

      // Allow higher utilization for top-level or large functions
      double maxUtilization = (funcName.contains("forward") && !funcName.contains("node"))
                              ? 3.0  // Allow 3x for top function
                              : (1.0 + balanceThreshold);

      if (utilizationRatio > maxUtilization) {
        continue; // Skip if too much resource usage
      }

      // Calculate score: balance resource utilization and communication cost
      double resourceScore = utilizationRatio;
      double communicationScore = 0.0;

      if (enableCommunicationOpt) {
        // Calculate communication cost
        uint64_t communicationCost = 0;
        for (auto callee : funcInfo.callees) {
          if (functionInfos.count(callee) &&
              functionInfos[callee].partitionId != -1 &&
              functionInfos[callee].partitionId != partitionId) {
            communicationCost += 1; // Inter-partition communication penalty
          }
        }
        for (auto caller : funcInfo.callers) {
          if (functionInfos.count(caller) &&
              functionInfos[caller].partitionId != -1 &&
              functionInfos[caller].partitionId != partitionId) {
            communicationCost += 1; // Inter-partition communication penalty
          }
        }
        communicationScore = (double)communicationCost / 10.0; // Normalize
      }

      double totalScore = resourceScore + communicationScore;

      if (totalScore < bestScore) {
        bestScore = totalScore;
        bestPartition = partitionId;
      }
    }

    // If no suitable partition found within constraints, use the least full partition
    if (bestPartition == -1) {
      llvm::errs() << "=== LAYOUT PARTITION DEBUG: No partition within constraints for "
                   << funcName << ", using least full partition " << leastFullPartition << "\n";
      bestPartition = leastFullPartition;
    }

    return bestPartition;
  }

  /// Step 4: Annotate functions with partition IDs
  void annotateFunctions(ModuleOp module) {
    llvm::errs() << "Annotating functions with partition IDs...\n";

    MLIRContext *ctx = module.getContext();
    auto i32Type = IntegerType::get(ctx, 32);

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isDeclaration()) continue;

      auto it = functionInfos.find(func.getName());
      if (it != functionInfos.end() && it->second.partitionId != -1) {
        func->setAttr("partition", IntegerAttr::get(i32Type, it->second.partitionId));

        llvm::errs() << "Annotated " << func.getName()
                     << " with partition " << it->second.partitionId << "\n";
      }
    }
  }

  /// Step 5: Generate Vivado floorplan constraint TCL script
  void generateTclScript(ModuleOp module) {
    llvm::errs() << "Generating Vivado floorplan constraint TCL: " << tclOutputPath << "\n";

    std::error_code EC;
    llvm::raw_fd_ostream tclFile(tclOutputPath, EC);
    if (EC) {
      llvm::errs() << "=== LAYOUT PARTITION DEBUG: Failed to create TCL file: " << EC.message() << "\n";
      return;
    }

    // Generate Vivado floorplan constraints
    generateVivadoFloorplanConstraints(tclFile);

    tclFile.close();
    llvm::errs() << "Vivado floorplan constraint TCL generated successfully\n";
  }

  /// Generate Vivado floorplan constraints for P&R
  void generateVivadoFloorplanConstraints(llvm::raw_fd_ostream &tclFile) {
    tclFile << "# =============================================================================\n";
    tclFile << "# Vivado Physical Floorplan Constraints\n";
    tclFile << "# Generated from HIDA Layout Partitioning\n";
    tclFile << "# Target Device: xcvu37p-fsvh2892-2L-e (Alveo U250)\n";
    tclFile << "# Number of partitions: " << numTiles << "\n";
    tclFile << "# =============================================================================\n\n";

    // -------------------------------------------------------------------------
    // Smart cell selection helper (TCL proc)
    // -------------------------------------------------------------------------
    tclFile << "proc get_func_cells {funcName} {\n";
    tclFile << "    set cells [list]\n";
    tclFile << "\n";
    tclFile << "    set c3 [get_cells -hier -quiet -filter \"NAME =~ *${funcName}*\"]\n";
    tclFile << "    if {[llength $c3] > 0} {\n";
    tclFile << "        lappend cells {*}$c3\n";
    tclFile << "    }\n";
    tclFile << "\n";
    tclFile << "    # 4) Deduplicate\n";
    tclFile << "    if {[llength $cells] == 0} {\n";
    tclFile << "        return {}\n";
    tclFile << "    }\n";
    tclFile << "    return [lsort -unique $cells]\n";
    tclFile << "}\n\n";

    // Add resource information as comments
    tclFile << "# Partition Resource Requirements:\n";
    for (int i = 0; i < numTiles; ++i) {
      auto &res = partitionResources[i];
      tclFile << "# Partition " << i << ": LUT=" << res.lut
              << ", FF=" << res.ff << ", DSP=" << res.dsp
              << ", BRAM=" << res.bram << ", URAM=" << res.uram << "\n";
    }
    tclFile << "\n";

    // Group functions by partition, skipping tiny/likely-inlined functions
    SmallVector<SmallVector<StringRef>> partitionFunctions(numTiles);

    // Threshold for "likely inlined" small logic-only functions
    const uint64_t LUT_INLINE_THRESHOLD = 50;

    for (auto &entry : functionInfos) {
      StringRef funcName = entry.first();
      auto &info = entry.second;

      if (info.partitionId < 0 || info.partitionId >= numTiles)
        continue;

      bool likelyInlined =
          info.resource.lut < LUT_INLINE_THRESHOLD &&
          info.resource.dsp == 0 &&
          info.resource.bram == 0;

      // These small functions still contributed to partitionResources,
      // but we don't explicitly floorplan them by name.
      if (likelyInlined)
        continue;

      partitionFunctions[info.partitionId].push_back(funcName);
    }

    tclFile << "# Remove any existing pblocks\n";
    tclFile << "catch {delete_pblocks [get_pblocks]}\n\n";

    // Define SLR regions for U250
    SmallVector<std::string> slrRegions = {"SLR0", "SLR1", "SLR2", "SLR3"};
    SmallVector<std::string> sliceRegions = {
      "SLICE_X0Y0:SLICE_X107Y119",
      "SLICE_X0Y120:SLICE_X107Y239",
      "SLICE_X0Y240:SLICE_X107Y359",
      "SLICE_X0Y360:SLICE_X107Y479"
    };
    SmallVector<std::string> dspRegions = {
      "DSP48E2_X0Y0:DSP48E2_X17Y47",
      "DSP48E2_X0Y48:DSP48E2_X17Y95",
      "DSP48E2_X0Y96:DSP48E2_X17Y143",
      "DSP48E2_X0Y144:DSP48E2_X17Y191"
    };
    SmallVector<std::string> bramRegions = {
      "RAMB36_X0Y0:RAMB36_X7Y23 RAMB18_X0Y0:RAMB18_X15Y47",
      "RAMB36_X0Y24:RAMB36_X7Y47 RAMB18_X0Y48:RAMB18_X15Y95",
      "RAMB36_X0Y48:RAMB36_X7Y71 RAMB18_X0Y96:RAMB18_X15Y143",
      "RAMB36_X0Y72:RAMB36_X7Y95 RAMB18_X0Y144:RAMB18_X15Y191"
    };

    // Generate pblock creation for each partition
    for (int i = 0; i < numTiles; ++i) {
      int slrIndex = i % 4;
      auto &res = partitionResources[i];

      tclFile << "# =============================================================================\n";
      tclFile << "# Partition " << i << " - " << slrRegions[slrIndex] << "\n";
      tclFile << "# Resources: DSP=" << res.dsp << ", BRAM=" << res.bram
              << ", LUT=" << res.lut << ", FF=" << res.ff << "\n";
      tclFile << "# =============================================================================\n";
      tclFile << "create_pblock pblock_partition_" << i << "\n";
      tclFile << "resize_pblock pblock_partition_" << i << " -add {" << slrRegions[slrIndex] << "}\n";
      tclFile << "set_property CONTAIN_ROUTING true [get_pblocks pblock_partition_" << i << "]\n";
      tclFile << "set_property SNAPPING_MODE ON [get_pblocks pblock_partition_" << i << "]\n";

      // Add resource-specific regions
      if (res.lut > 0 || res.ff > 0) {
        tclFile << "resize_pblock pblock_partition_" << i << " -add {" << sliceRegions[slrIndex] << "}\n";
      }
      if (res.dsp > 0) {
        tclFile << "resize_pblock pblock_partition_" << i << " -add {" << dspRegions[slrIndex] << "}\n";
      }
      if (res.bram > 0) {
        tclFile << "resize_pblock pblock_partition_" << i << " -add {" << bramRegions[slrIndex] << "}\n";
      }

      // Add cells to pblock based on function names (via helper proc)
      tclFile << "\n# Add cells for partition " << i << " functions\n";
      tclFile << "set partition_" << i << "_cells [list]\n";
      for (auto funcName : partitionFunctions[i]) {
        tclFile << "set _cells [get_func_cells " << funcName << "]\n";
        tclFile << "if {[llength $_cells] > 0} { "
                << "lappend partition_" << i << "_cells $_cells }\n";
      }
      tclFile << "if {[llength $partition_" << i << "_cells] > 0} {\n";
      tclFile << "    add_cells_to_pblock pblock_partition_" << i
              << " $partition_" << i << "_cells\n";
      tclFile << "    puts \"Added [llength $partition_" << i
              << "_cells] cells to pblock_partition_" << i << "\"\n";
      tclFile << "}\n\n";
    }

    // Add inter-SLR routing optimization
    tclFile << "# =============================================================================\n";
    tclFile << "# Inter-SLR Routing Optimization\n";
    tclFile << "# =============================================================================\n";
    tclFile << "set_property ROUTE.SLR_CROSSING_MINIMIZATION true [current_design]\n";
    tclFile << "set_property ROUTE.SLR_CROSSING_OPTIMIZATION true [current_design]\n";
    tclFile << "set_property MAX_FANOUT 50 [get_nets -hierarchical]\n\n";

    tclFile << "puts \"=== Floorplan constraints applied successfully ===\"\n";
  }

  /// Debug: Print partitioning results
  void printPartitioningResults() {
    llvm::errs() << "\n=== LAYOUT PARTITION DEBUG: Partitioning Results ===\n";
    for (int i = 0; i < numTiles; ++i) {
      llvm::errs() << "=== LAYOUT PARTITION DEBUG: Partition " << i << ":\n";
      auto &res = partitionResources[i];
      llvm::errs() << "=== LAYOUT PARTITION DEBUG:   Resources: LUT=" << res.lut << ", FF=" << res.ff
                   << ", DSP=" << res.dsp << ", BRAM=" << res.bram
                   << ", URAM=" << res.uram << "\n";
      llvm::errs() << "=== LAYOUT PARTITION DEBUG:   Functions: ";
      for (auto &entry : functionInfos) {
        StringRef funcName = entry.first();
        auto &info = entry.second;
        if (info.partitionId == i) {
          llvm::errs() << funcName << " ";
        }
      }
      llvm::errs() << "\n\n";
    }
  }
};

namespace mlir {
namespace scalehls {

std::unique_ptr<Pass> createLayoutPartitionPass(int numTiles,
                                                std::string tclOutput,
                                                double balanceThreshold,
                                                bool enableCommOpt) {
  auto pass = std::make_unique<LayoutPartitionPass>();
  pass->numTiles = numTiles;
  pass->tclOutputPath = tclOutput;
  pass->balanceThreshold = balanceThreshold;
  pass->enableCommunicationOpt = enableCommOpt;
  return pass;
}

// Overload for backward compatibility (standalone usage)
std::unique_ptr<Pass> createLayoutPartitionPass() {
  return std::make_unique<LayoutPartitionPass>();
}

} // namespace scalehls
} // namespace mlir
