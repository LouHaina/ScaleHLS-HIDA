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

namespace {

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

  /// Constructor with parameters
  LayoutPartitionPass() = default;
  LayoutPartitionPass(int tiles, std::string tclOutput, double threshold, bool commOpt)
      : numTiles(tiles), tclOutputPath(tclOutput), balanceThreshold(threshold),
        enableCommunicationOpt(commOpt) {}

  /// Command line options (for standalone usage)
  Option<int> numTiles{*this, "tiles", llvm::cl::init(2),
                       llvm::cl::desc("Number of FPGA tiles/partitions")};
  
  Option<std::string> tclOutputPath{*this, "tcl-output", llvm::cl::init("partition.tcl"),
                                    llvm::cl::desc("Output path for TCL script")};
  
  Option<double> balanceThreshold{*this, "balance-threshold", llvm::cl::init(0.3),
                                  llvm::cl::desc("Resource balance threshold (0.0-1.0)")};

  Option<bool> enableCommunicationOpt{*this, "enable-comm-opt", llvm::cl::init(true),
                                      llvm::cl::desc("Enable communication optimization")};

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
    
    // Step 5: Generate TCL script for Vitis HLS
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
    LLVM_DEBUG(llvm::dbgs() << "Collecting function information...\n");
    
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
        
        LLVM_DEBUG(llvm::dbgs() << "Function " << func.getName() 
                   << ": LUT=" << info.resource.lut 
                   << ", DSP=" << info.resource.dsp 
                   << ", BRAM=" << info.resource.bram << "\n");
      } else {
        // Fallback: estimate resources based on function complexity
        info.resource.lut = 100;  // Default estimate
        info.resource.ff = 100;
        info.resource.dsp = 1;
        info.resource.bram = 1;
        
        LLVM_DEBUG(llvm::dbgs() << "Function " << func.getName() 
                   << ": Using default resource estimates\n");
      }
      
      // Collect function call relationships
      llvm::SetVector<StringRef> callees;
      func.walk([&](func::CallOp call) {
        callees.insert(call.getCallee());
      });
      info.callees.append(callees.begin(), callees.end());
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Total resources: LUT=" << totalResources.lut 
               << ", DSP=" << totalResources.dsp 
               << ", BRAM=" << totalResources.bram << "\n");
    
    return success();
  }

  /// Step 2: Build bidirectional communication graph
  void buildCommunicationGraph() {
    LLVM_DEBUG(llvm::dbgs() << "Building communication graph...\n");
    
    // Build caller-callee relationships
    for (auto &[funcName, info] : functionInfos) {
      for (auto callee : info.callees) {
        if (functionInfos.count(callee)) {
          functionInfos[callee].callers.push_back(funcName);
        }
      }
    }
    
    // Calculate communication weights (simplified model)
    for (auto &[funcName, info] : functionInfos) {
      info.communicationWeight = info.callees.size() + info.callers.size();
    }
  }

  /// Step 3: Resource-aware partitioning algorithm
  LogicalResult runResourceAwarePartitioning() {
    LLVM_DEBUG(llvm::dbgs() << "Running resource-aware partitioning...\n");
    
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
    for (auto &[funcName, info] : functionInfos) {
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
        LLVM_DEBUG(llvm::dbgs() << "Failed to find suitable partition for " 
                   << funcName << "\n");
        return failure();
      }
      
      funcInfo.partitionId = bestPartition;
      partitionResources[bestPartition] += funcInfo.resource;
      
      LLVM_DEBUG(llvm::dbgs() << "Assigned " << funcName 
                 << " to partition " << bestPartition << "\n");
    }
    
    return success();
  }

  /// Find the best partition for a function considering resources and communication
  int findBestPartition(StringRef funcName, const HardwareResource &targetPerPartition) {
    auto &funcInfo = functionInfos[funcName];
    int bestPartition = -1;
    double bestScore = std::numeric_limits<double>::max();
    
    for (int partitionId = 0; partitionId < numTiles; ++partitionId) {
      auto &partitionRes = partitionResources[partitionId];
      auto newPartitionRes = partitionRes + funcInfo.resource;
      
      // Check if resources fit (with some tolerance)
      double utilizationRatio = newPartitionRes.getUtilizationRatio(targetPerPartition);
      if (utilizationRatio > (1.0 + balanceThreshold)) {
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
    
    return bestPartition;
  }

  /// Step 4: Annotate functions with partition IDs
  void annotateFunctions(ModuleOp module) {
    LLVM_DEBUG(llvm::dbgs() << "Annotating functions with partition IDs...\n");
    
    MLIRContext *ctx = module.getContext();
    auto i32Type = IntegerType::get(ctx, 32);
    
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isDeclaration()) continue;
      
      auto it = functionInfos.find(func.getName());
      if (it != functionInfos.end() && it->second.partitionId != -1) {
        func->setAttr("partition", IntegerAttr::get(i32Type, it->second.partitionId));
        
        LLVM_DEBUG(llvm::dbgs() << "Annotated " << func.getName() 
                   << " with partition " << it->second.partitionId << "\n");
      }
    }
  }

  /// Step 5: Generate TCL script for Vitis HLS synthesis
  void generateTclScript(ModuleOp module) {
    LLVM_DEBUG(llvm::dbgs() << "Generating TCL script: " << tclOutputPath << "\n");
    
    std::error_code EC;
    llvm::raw_fd_ostream tclFile(tclOutputPath, EC);
    if (EC) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to create TCL file: " << EC.message() << "\n");
      return;
    }
    
    tclFile << "# Generated TCL script for HIDA Layout Partitioning\n";
    tclFile << "# Number of partitions: " << numTiles << "\n\n";
    
    // Group functions by partition
    SmallVector<SmallVector<StringRef>> partitionFunctions(numTiles);
    for (auto &[funcName, info] : functionInfos) {
      if (info.partitionId >= 0 && info.partitionId < numTiles) {
        partitionFunctions[info.partitionId].push_back(funcName);
      }
    }
    
    // Generate partition directives
    for (int i = 0; i < numTiles; ++i) {
      tclFile << "# Partition " << i << " functions:\n";
      for (auto funcName : partitionFunctions[i]) {
        tclFile << "set_directive_allocation -limit 1 -type function " 
                << funcName << " -partition " << i << "\n";
      }
      tclFile << "\n";
    }
    
    // Add resource utilization comments
    tclFile << "# Resource utilization per partition:\n";
    for (int i = 0; i < numTiles; ++i) {
      auto &res = partitionResources[i];
      tclFile << "# Partition " << i << ": LUT=" << res.lut 
              << ", FF=" << res.ff << ", DSP=" << res.dsp 
              << ", BRAM=" << res.bram << ", URAM=" << res.uram << "\n";
    }
    
    tclFile.close();
    LLVM_DEBUG(llvm::dbgs() << "TCL script generated successfully\n");
  }

  /// Debug: Print partitioning results
  void printPartitioningResults() {
    LLVM_DEBUG({
      llvm::dbgs() << "\n=== Partitioning Results ===\n";
      for (int i = 0; i < numTiles; ++i) {
        llvm::dbgs() << "Partition " << i << ":\n";
        auto &res = partitionResources[i];
        llvm::dbgs() << "  Resources: LUT=" << res.lut << ", FF=" << res.ff 
                     << ", DSP=" << res.dsp << ", BRAM=" << res.bram 
                     << ", URAM=" << res.uram << "\n";
        llvm::dbgs() << "  Functions: ";
        for (auto &[funcName, info] : functionInfos) {
          if (info.partitionId == i) {
            llvm::dbgs() << funcName << " ";
          }
        }
        llvm::dbgs() << "\n\n";
      }
    });
  }
};

} // namespace

namespace scalehls {

std::unique_ptr<Pass> createLayoutPartitionPass(int numTiles,
                                                std::string tclOutput,
                                                double balanceThreshold,
                                                bool enableCommOpt) {
  return std::make_unique<LayoutPartitionPass>(numTiles, tclOutput, balanceThreshold, enableCommOpt);
}

// Overload for backward compatibility (standalone usage)
std::unique_ptr<Pass> createLayoutPartitionPass() {
  return std::make_unique<LayoutPartitionPass>();
}

} // namespace scalehls