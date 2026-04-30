//===- LayoutPartitionEnhanced.cpp - Enhanced Layout Partitioning -*- C++ -*-===//
//
// Enhanced layout-driven partitioning optimization pass for HIDA.
// Implements FADO/Autobridge-style partitioning (Connectivity-driven BFS + FM Refinement).
//
// Key behaviors:
//  - Initial Placement: Uses BFS starting from Top function to cluster connected logic.
//  - Refinement: Uses Iterative Fiduccia-Mattheyses (FM) moves to minimize SLL usage.
//  - Edge Weights: Calculated based on bitwidth of CallOp arguments.
//  - Top-level function acts as the anchor but is NOT assigned to any partition.
//  - Vivado TCL uses only SLR-level pblocks: resize_pblock ... -add {SLR0}.
//
// Device model:
//   Target FPGA: XCVU37P (U280/U250-class, ~3 SLRs)
//
//
// NOTE: BRAM handling has been adjusted because QoR-estimated BRAM is
//       significantly higher than the actual HLS implementation result.
//       We now effectively ignore BRAM in partitioning decisions.
//
//===----------------------------------------------------------------------===//

#include "scalehls/Transforms/Passes.h"
#include "scalehls/Dialect/HLS/HLS.h"
#include "scalehls/Dialect/HLS/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <deque>
#include <set>
#include <map>

#define DEBUG_TYPE "scalehls-layout-partition"

using namespace mlir;
using namespace scalehls;
using namespace hls;

namespace {

/// Hardware resource information for a function or partition.
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
    lut  += other.lut;
    ff   += other.ff;
    dsp  += other.dsp;
    bram += other.bram;
    uram += other.uram;
    return *this;
  }

  HardwareResource operator+(const HardwareResource &other) const {
    HardwareResource tmp = *this;
    tmp += other;
    return tmp;
  }

  // Added for FM Refinement (removing a node from a partition)
  HardwareResource operator-(const HardwareResource &other) const {
    HardwareResource tmp;
    tmp.lut  = (lut > other.lut)   ? lut - other.lut : 0;
    tmp.ff   = (ff > other.ff)     ? ff - other.ff : 0;
    tmp.dsp  = (dsp > other.dsp)   ? dsp - other.dsp : 0;
    tmp.bram = (bram > other.bram) ? bram - other.bram : 0;
    tmp.uram = (uram > other.uram) ? uram - other.uram : 0;
    return tmp;
  }

  /// Max utilization ratio vs. a given "capacity" constraint.
  double getUtilizationRatio(const HardwareResource &cap) const {
    double maxRatio = 0.0;
    if (cap.lut  > 0) maxRatio = std::max(maxRatio, (double)lut  / cap.lut);
    if (cap.ff   > 0) maxRatio = std::max(maxRatio, (double)ff   / cap.ff);
    if (cap.dsp  > 0) maxRatio = std::max(maxRatio, (double)dsp  / cap.dsp);
    if (cap.bram > 0) maxRatio = std::max(maxRatio, (double)bram / cap.bram);
    if (cap.uram > 0) maxRatio = std::max(maxRatio, (double)uram / cap.uram);
    return maxRatio;
  }

  /// Rough "weight" for sorting (DSP/BRAM weighted higher).
  /// BRAM is removed here because QoR-estimated BRAM is unreliable
  /// for current HIDA flows (actual BRAM usage is ~0).
  uint64_t getTotalWeight() const {
    return lut + ff + dsp * 10 /* + bram * 8 */ + uram * 12;
  }
};

/// Per-function information used for partitioning.
struct FunctionInfo {
  HardwareResource resource;
  SmallVector<StringRef, 4> callees;
  SmallVector<StringRef, 4> callers;
  int partitionId = -1;
  uint64_t communicationWeight = 0; // Total connected bitwidth
  bool skipFloorplan = false; // e.g., tiny helper or top-level.
};

/// Layout partitioning pass.
struct LayoutPartitionPass
    : public PassWrapper<LayoutPartitionPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutPartitionPass)

  LayoutPartitionPass() = default;

  // Public parameters
  int numTiles = 2;
  std::string tclOutputPath = "floorplan_constraints.tcl";
  double balanceThreshold = 0.3;
  bool enableCommunicationOpt = true;
  double singleSlrUtilThreshold = 0.8;
  uint64_t lutInlineThreshold = 50;

  StringRef getArgument() const override {
    return "scalehls-layout-partition";
  }

  StringRef getDescription() const override {
    return "Enhanced layout-driven partitioning for HIDA (FADO/Autobridge Style)";
  }

  void runOnOperation() override {
    if (numTiles <= 0)
      numTiles = 1;

    ModuleOp module = getOperation();
    if (failed(collectFunctionInfo(module))) {
      signalPassFailure();
      return;
    }

    identifyTopFunction();

    // Decide whether to partition at all.
    bool keepSingleTile = shouldKeepSingleTile();
    if (keepSingleTile) {
      llvm::errs() << "=== LAYOUT PARTITION: Design small enough, "
                      "keeping everything in a single SLR (partition 0)\n";
      numTiles = 1;

      // Assign partitionId = 0 to all non-top functions.
      // Keep top unassigned (partitionId = -1).
      for (auto &entry : functionInfos) {
        if (entry.first().str() == topFunctionName) {
          entry.second.partitionId = -1;
        } else {
          entry.second.partitionId = 0;
        }
      }

      // Recompute resources from final partition assignments.
      recomputePartitionResources();
    } else {
      // 1. Initial Solution: BFS Cluster Growth (capacity-aware)
      if (failed(runBFSInitialPartitioning())) {
        llvm::errs() << "=== LAYOUT PARTITION: Initial BFS placement failed "
                        "due to hard capacity constraints. ===\n";
        signalPassFailure();
        return;
      }
      // 2. Refinement: Iterative Gain-Based Moving (Autobridge/FM)
      optimizePartitions();

      // Recompute resources from the final partition assignments.
      recomputePartitionResources();
    }

    // Final hard capacity check for ALL SLRs. If violated, fail the pass.
    if (!checkHardCapacity("final partitioning")) {
      signalPassFailure();
      return;
    }

    annotateFunctions(module);
    generateTclScript();
    printPartitioningResults();
  }

private:
  llvm::StringMap<FunctionInfo> functionInfos;
  SmallVector<HardwareResource> partitionResources;
  HardwareResource totalResources;
  std::string topFunctionName; // e.g. "forward"

  // Track specific edge weights (Bitwidth) between functions: {A, B} -> bits
  // Key is sorted pair of strings to make it undirected.
  std::map<std::pair<std::string, std::string>, int> edgeWeights;

  // Recompute partitionResources from final partitionId assignments.
  void recomputePartitionResources() {
    partitionResources.assign(numTiles, HardwareResource());
    for (auto &entry : functionInfos) {
      int pid = entry.second.partitionId;
      if (pid >= 0 && pid < numTiles)
        partitionResources[pid] += entry.second.resource;
    }
  }

  // Hard capacity check: no partition may exceed X% of SLR capacity.
  bool checkHardCapacity(StringRef stage) {
    HardwareResource cap = getPartitionCapacity();
    bool ok = true;

    for (int i = 0; i < numTiles; ++i) {
      if (i >= (int)partitionResources.size())
        break;

      double util = partitionResources[i].getUtilizationRatio(cap);
      if (util > singleSlrUtilThreshold + 1e-6) {
        llvm::errs() << "ERROR: " << stage
                     << ": Partition " << i
                     << " exceeds per-SLR utilization limit "
                     << singleSlrUtilThreshold
                     << " (util=" << util << ")\n";
        ok = false;
      }
    }

    if (!ok) {
      llvm::errs()
          << "=== LAYOUT PARTITION: Hard capacity constraint violated; "
             "no valid floorplan can be constructed. ===\n";
    }
    return ok;
  }

//===--------------------------------------------------------------------===//
  // Step 1: Collect Info & Calculate Bitwidth Weights
  //===--------------------------------------------------------------------===//

  LogicalResult collectFunctionInfo(ModuleOp module) {
    llvm::errs() << "=== LAYOUT PARTITION: Collecting function information...\n";

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isDeclaration()) continue;

      StringRef funcName = func.getName();
      auto &info = functionInfos[funcName];

      // Resource Estimation
      if (auto resAttr = getResource(func)) {
        info.resource.lut  = resAttr.getLut();
        info.resource.dsp  = resAttr.getDsp();
        info.resource.ff   = info.resource.lut; // approx
        info.resource.uram = 0;

        // BRAM from the QoR estimation is very inaccurate for the current
        // HIDA + Vitis HLS flow. Actual HLS reports show almost 0 BRAM
        // usage, while the estimator may report large values (e.g., 272).
        // To avoid over-constraining the layout, we treat BRAM as 0.
        info.resource.bram = 0;
      } else {
        info.resource.lut  = 100;
        info.resource.ff   = 100;
        info.resource.dsp  = 1;
        info.resource.bram = 0;
        info.resource.uram = 0;
      }
      totalResources += info.resource;

      // Build Graph & Calculate Bitwidth Weights (Autobridge logic)
      llvm::SetVector<StringRef> uniqueCallees;
      func.walk([&](func::CallOp call) {
        StringRef calleeName = call.getCallee();
        uniqueCallees.insert(calleeName);

        // Calculate Bitwidth weight safely
        int callWeight = 0;
        for (auto arg : call.getOperands()) {
          Type t = arg.getType();

          if (t.isIntOrFloat()) {
            callWeight += t.getIntOrFloatBitWidth();
          } else if (t.isIndex()) {
            callWeight += 64;
          } else if (auto memref = t.dyn_cast<MemRefType>()) {
            Type elType = memref.getElementType();
            if (elType.isIntOrFloat())
              callWeight += elType.getIntOrFloatBitWidth();
            else
              callWeight += 64;
          } else {
            callWeight += 64;
          }
        }

        if (callWeight == 0) callWeight = 1; // Control signal baseline

        // Store edge weight undirected
        std::string a = funcName.str();
        std::string b = calleeName.str();
        if (a > b) std::swap(a, b);
        edgeWeights[{a, b}] += callWeight;
      });

      info.callees.append(uniqueCallees.begin(), uniqueCallees.end());
    }

    // Back-fill callers
    for (auto &entry : functionInfos) {
      StringRef funcName = entry.first();
      for (auto callee : entry.second.callees) {
        if (functionInfos.count(callee)) {
          functionInfos[callee].callers.push_back(funcName);
        }
      }
    }

    // Calculate node communication weight (sum of all incident edges)
    for (auto &entry : functionInfos) {
      StringRef name = entry.first();
      int totalW = 0;
      auto sumW = [&](StringRef neighbor) {
        std::string a = name.str();
        std::string b = neighbor.str();
        if (a > b) std::swap(a, b);
        totalW += edgeWeights[{a, b}];
      };
      for (auto c : entry.second.callees) sumW(c);
      for (auto c : entry.second.callers) sumW(c);
      entry.second.communicationWeight = totalW;
    }

    llvm::errs() << "=== LAYOUT PARTITION: Total resources: LUT=" << totalResources.lut
                 << ", DSP=" << totalResources.dsp
                 << ", BRAM=" << totalResources.bram << "\n";
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Step 2: Helpers
  //===--------------------------------------------------------------------===//

  void identifyTopFunction() {
    // Prefer "forward" as explicit top.
    for (auto &entry : functionInfos) {
      if (entry.first().equals("forward")) {
        topFunctionName = entry.first().str();
        entry.second.skipFloorplan = true;  // Explicitly mark as non-floorplanned
        return;
      }
    }
    // Fallback: function with no callers and largest resource.
    uint64_t maxRes = 0;
    for (auto &entry : functionInfos) {
      if (entry.second.callers.empty() &&
          entry.second.resource.getTotalWeight() > maxRes) {
        maxRes = entry.second.resource.getTotalWeight();
        topFunctionName = entry.first().str();
      }
    }
    if (!topFunctionName.empty()) {
      llvm::errs() << "=== LAYOUT PARTITION: Top function detected: "
                   << topFunctionName << "\n";
      auto it = functionInfos.find(topFunctionName);
      if (it != functionInfos.end())
        it->second.skipFloorplan = true;
    }
  }

  bool isTopFunction(StringRef name) const {
    return !topFunctionName.empty() && name == topFunctionName;
  }

  bool shouldKeepSingleTile() const {
    if (numTiles <= 1) return true;
    HardwareResource slrCap(434560, 869120, 3008, 1100, 500);
    double util = totalResources.getUtilizationRatio(slrCap);
    llvm::errs() << "=== LAYOUT PARTITION: single-SLR utilization ~= "
                 << util << " (threshold " << singleSlrUtilThreshold << ")\n";
    return util < singleSlrUtilThreshold;
  }

  // Get capacity for a single partition
  HardwareResource getPartitionCapacity() {
    // XCVU37P approx per-SLR capacity.
    // BRAM is set to 0 so that it is ignored by getUtilizationRatio()
    // (since cap.bram == 0 → that dimension is skipped).
    return HardwareResource(434560, 869120, 3008, 0, 500);
  }

  //===--------------------------------------------------------------------===//
  // Step 3: BFS Initial Partitioning (Capacity-Aware + Top Exclusion)
  //===--------------------------------------------------------------------===//

  LogicalResult runBFSInitialPartitioning() {
    llvm::errs() << "=== LAYOUT PARTITION: Running BFS Initial Placement...\n";
    partitionResources.assign(numTiles, HardwareResource());
    HardwareResource cap = getPartitionCapacity();

    std::set<StringRef> visited;
    std::deque<StringRef> queue;

    // Start BFS from top function if available.
    if (!topFunctionName.empty()) {
      queue.push_back(StringRef(topFunctionName));
      visited.insert(StringRef(topFunctionName));
    }

    // Enqueue any remaining functions (disconnected components).
    for (auto &entry : functionInfos) {
      if (visited.find(entry.first()) == visited.end())
        queue.push_back(entry.first());
    }

    while (!queue.empty()) {
      StringRef currName = queue.front();
      queue.pop_front();

      auto &info = functionInfos[currName];

      // Never assign the top function to a partition.
      if (isTopFunction(currName)) {
        info.partitionId = -1;
      } else {
        // HARD capacity rule: pick a partition where util <= singleSlrUtilThreshold
        int bestP = -1;
        double bestUtil = std::numeric_limits<double>::max();

        for (int p = 0; p < numTiles; ++p) {
          double utilIf =
              (partitionResources[p] + info.resource).getUtilizationRatio(cap);
          if (utilIf <= singleSlrUtilThreshold && utilIf < bestUtil) {
            bestUtil = utilIf;
            bestP = p;
          }
        }

        if (bestP < 0) {
          llvm::errs() << "ERROR: BFS placement cannot assign function '"
                       << currName
                       << "' to any partition without exceeding per-SLR "
                          "utilization limit "
                       << singleSlrUtilThreshold << ".\n";
          return failure();
        }

        info.partitionId = bestP;
        partitionResources[bestP] += info.resource;
      }

      auto addNeighbor = [&](StringRef n) {
        if (visited.find(n) == visited.end()) {
          visited.insert(n);
          queue.push_back(n);
        }
      };

      auto &callees = info.callees;
      std::sort(callees.begin(), callees.end(), [&](StringRef a, StringRef b) {
        std::string sA = currName.str(), sB = currName.str();
        std::string nA = a.str(), nB = b.str();
        if (sA > nA) std::swap(sA, nA);
        if (sB > nB) std::swap(sB, nB);
        return edgeWeights[{sA, nA}] > edgeWeights[{sB, nB}];
      });

      for (auto c : callees) addNeighbor(c);
      for (auto c : info.callers) addNeighbor(c);
    }
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Step 4: Iterative Refinement (Fiduccia-Mattheyses Style)
  // Attempts to move nodes to neighbors to minimize SLL crossing cost.
  //===--------------------------------------------------------------------===//

  void optimizePartitions() {
    llvm::errs() << "=== LAYOUT PARTITION: Running FM Iterative Refinement...\n";
    HardwareResource cap = getPartitionCapacity();

    int maxPasses = 20;
    bool improved = true;

    for (int pass = 0; pass < maxPasses && improved; ++pass) {
      improved = false;
      int moves = 0;

      // Sort candidates by connectivity weight (heuristic order)
      SmallVector<StringRef, 32> candidates;
      for (auto &entry : functionInfos) candidates.push_back(entry.first());

      llvm::sort(candidates, [&](StringRef a, StringRef b) {
        return functionInfos[a].communicationWeight > functionInfos[b].communicationWeight;
      });

      for (StringRef name : candidates) {
        // Never move the top function.
        if (isTopFunction(name))
          continue;

        auto &info = functionInfos[name];
        int currentP = info.partitionId;
        if (currentP == -1) continue;

        // Check adjacent partitions (e.g. 0 <-> 1, 1 <-> 2)
        SmallVector<int, 2> adj;
        if (currentP > 0) adj.push_back(currentP - 1);
        if (currentP < numTiles - 1) adj.push_back(currentP + 1);

        int bestDest = -1;
        int maxGain = 0; // Must be positive to move

        for (int targetP : adj) {
          // 1. Capacity Check (Hard constraint)
          auto newRes = partitionResources[targetP] + info.resource;
          if (newRes.getUtilizationRatio(cap) > singleSlrUtilThreshold) continue;

          // 2. Gain Calculation
          int gain = calculateMoveGain(name, currentP, targetP);
          if (gain > maxGain) {
            maxGain = gain;
            bestDest = targetP;
          }
        }

        if (bestDest != -1) {
          // Perform Move
          partitionResources[currentP] = partitionResources[currentP] - info.resource;
          partitionResources[bestDest] += info.resource;
          info.partitionId = bestDest;
          improved = true;
          moves++;
        }
      }
      llvm::errs() << "   Pass " << pass << ": Moved " << moves << " functions.\n";
    }
  }

  // Gain = (Wires Removed) - (Wires Added)
  int calculateMoveGain(StringRef name, int currentP, int targetP) {
    int internalCost = 0; // Edges to currentP (would become Cuts if we move)
    int externalCost = 0; // Edges to targetP (would become Internal if we move)

    auto accumulate = [&](StringRef neighbor) {
      if (neighbor == name) return;
      if (functionInfos.find(neighbor) == functionInfos.end()) return;

      int neighborP = functionInfos[neighbor].partitionId;

      // Get weight
      std::string a = name.str();
      std::string b = neighbor.str();
      if (a > b) std::swap(a, b);
      int w = edgeWeights[{a, b}];

      if (neighborP == currentP) internalCost += w; // Adding a cut
      if (neighborP == targetP)  externalCost += w; // Removing a cut
    };

    auto &info = functionInfos[name];
    for (auto c : info.callees) accumulate(c);
    for (auto c : info.callers) accumulate(c);

    return externalCost - internalCost;
  }

  //===--------------------------------------------------------------------===//
  // Step 5: Annotate MLIR functions with partition IDs
  //===--------------------------------------------------------------------===//

  void annotateFunctions(ModuleOp module) {
    llvm::errs() << "=== LAYOUT PARTITION: Annotating MLIR functions with partition IDs...\n";
    MLIRContext *ctx = module.getContext();
    auto i32Type = IntegerType::get(ctx, 32);

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isDeclaration())
        continue;

      StringRef name = func.getName();
      auto it = functionInfos.find(name);
      if (it == functionInfos.end())
        continue;

      int pid = it->second.partitionId;
      if (pid < 0)
        continue; // top or unassigned

      func->setAttr("partition", IntegerAttr::get(i32Type, pid));
      llvm::errs() << "  " << name << " -> partition " << pid << "\n";
    }
  }

  //===--------------------------------------------------------------------===//
  // Step 6: Generate Vivado TCL
  //===--------------------------------------------------------------------===//

  void generateTclScript() {
    llvm::errs() << "=== LAYOUT PARTITION: Generating Vivado TCL: "
                 << tclOutputPath << "\n";

    std::error_code ec;
    llvm::raw_fd_ostream os(tclOutputPath, ec);
    if (ec) {
      llvm::errs() << "  ERROR: Cannot open TCL output file: "
                   << ec.message() << "\n";
      return;
    }

    emitVivadoFloorplanConstraints(os);
    os.close();
  }

  void emitVivadoFloorplanConstraints(llvm::raw_fd_ostream &tcl) {
    tcl << "# ============================================================================\n";
    tcl << "# Vivado SLR Floorplan Constraints (Generated by HIDA LayoutPartitionPass)\n";
    tcl << "# Target: xcvu37p (U250/U280 class)\n";
    tcl << "# Partitions: " << numTiles << "\n";
    tcl << "# NOTE: Constraints are SLR-only (no slice/DSP/BRAM subranges).\n";
    tcl << "# ============================================================================\n\n";

    // Helper proc to collect cells for a function name.
    tcl << "proc get_func_cells {funcName} {\n";
    tcl << "    set cells [list]\n\n";
    tcl << "    # Pattern 1: typical HLS instance, e.g. grp_forward_node82_fu_..._U0\n";
    tcl << "    set pat1 \"*${funcName}*_U0\"\n";
    tcl << "    set c [get_cells -hier -quiet -filter \"NAME =~ $pat1\"]\n\n";
    tcl << "    # If nothing, try a looser pattern\n";
    tcl << "    if {[llength $c] == 0} {\n";
    tcl << "        set pat2 \"*${funcName}*\"\n";
    tcl << "        set c [get_cells -hier -quiet -filter \"NAME =~ $pat2\"]\n";
    tcl << "        puts \"get_func_cells: $funcName -> [llength $c] cells (fallback pattern: $pat2)\"\n";
    tcl << "    } else {\n";
    tcl << "        puts \"get_func_cells: $funcName -> [llength $c] cells (pattern: $pat1)\"\n";
    tcl << "    }\n\n";
    tcl << "    # Safely accumulate (no {*} expansion)\n";
    tcl << "    foreach cell $c {\n";
    tcl << "        lappend cells $cell\n";
    tcl << "    }\n\n";
    tcl << "    return [lsort -unique $cells]\n";
    tcl << "}\n\n";

    // Partition resource summary (approximate, algorithm-side).
    tcl << "# Partition resource summary (approximate, algorithm-level):\n";
    for (int i = 0; i < numTiles; ++i) {
      auto &r = partitionResources[i];
      tcl << "#   Partition " << i << ": LUT=" << r.lut
          << ", FF="   << r.ff
          << ", DSP="  << r.dsp
          << ", BRAM=" << r.bram
          << ", URAM=" << r.uram << "\n";
    }
    tcl << "\n";

    // Group functions by partition, skipping tiny helpers and top-level.
    SmallVector<SmallVector<StringRef>> partFuncs(numTiles);
    for (auto &entry : functionInfos) {
      StringRef name = entry.first();
      auto &info = entry.second;

      if (info.partitionId < 0 || info.partitionId >= numTiles)
        continue;

      // Skip top function from explicit floorplan (already marked skipFloorplan).
      if (isTopFunction(name)) {
        info.skipFloorplan = true;
        continue;
      }

      bool tiny =
          info.resource.lut < lutInlineThreshold &&
          info.resource.dsp == 0 &&
          info.resource.bram == 0;
      if (tiny) {
        info.skipFloorplan = true;
        continue;
      }

      partFuncs[info.partitionId].push_back(name);
    }

    tcl << "# Remove existing pblocks to avoid conflicts.\n";
    tcl << "catch {delete_pblocks [get_pblocks]}\n\n";

    tcl << "# Create SLR-level pblocks per non-empty partition.\n";

    for (int i = 0; i < numTiles; ++i) {
      // Skip partitions that have no floorplanned functions.
      if (partFuncs[i].empty())
        continue;

      int slrIdx = i; // Direct mapping: partition i -> SLRi.
      tcl << "create_pblock pblock_partition_" << i << "\n";
      tcl << "resize_pblock pblock_partition_" << i
          << " -add {SLR" << slrIdx << "}\n";
      tcl << "set_property CONTAIN_ROUTING true [get_pblocks pblock_partition_" << i << "]\n";
      tcl << "set_property SNAPPING_MODE ON   [get_pblocks pblock_partition_" << i << "]\n\n";

      tcl << "# Add cells for partition " << i << "\n";
      tcl << "set partition_" << i << "_cells [list]\n\n";
      for (auto funcName : partFuncs[i]) {
        tcl << "set _cells [get_func_cells " << funcName << "]\n";
        tcl << "if {[llength $_cells] > 0} { set partition_" << i
            << "_cells [concat $partition_" << i << "_cells $_cells] }\n\n";
      }
      tcl << "if {[llength $partition_" << i << "_cells] > 0} {\n";
      tcl << "  add_cells_to_pblock pblock_partition_" << i
          << " $partition_" << i << "_cells\n";
      tcl << "  puts \"Added [llength $partition_" << i
          << "_cells] cells to pblock_partition_" << i << "\"\n";
      tcl << "}\n\n";
    }

    tcl << "puts \"=== HIDA SLR floorplan constraints applied ===\"\n";
  }

  //===--------------------------------------------------------------------===//
  // Debug printing
  //===--------------------------------------------------------------------===//

  void printPartitioningResults() {
    llvm::errs() << "\n=== LAYOUT PARTITION: Partitioning Results ===\n";
    for (int i = 0; i < numTiles; ++i) {
      llvm::errs() << "  Partition " << i << ":\n";
      auto &r = partitionResources[i];
      llvm::errs() << "    Resources: LUT=" << r.lut
                   << ", FF="   << r.ff
                   << ", DSP="  << r.dsp
                   << ", BRAM=" << r.bram
                   << ", URAM=" << r.uram << "\n";
      llvm::errs() << "    Functions: ";
      for (auto &entry : functionInfos) {
        auto &info = entry.second;
        if (info.partitionId == i)
          llvm::errs() << entry.first() << " ";
      }
      llvm::errs() << "\n";
    }
    llvm::errs() << "=== End of partitioning report ===\n";

    if (!topFunctionName.empty()) {
      llvm::errs() << "Top function (unpartitioned): " << topFunctionName << "\n";
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace scalehls {

std::unique_ptr<Pass> createLayoutPartitionPass(
    int numTiles,
    std::string tclOutput,
    double balanceThreshold,
    bool enableCommunicationOpt,
    double singleSlrUtilThreshold) {
  auto pass = std::make_unique<LayoutPartitionPass>();
  pass->numTiles           = numTiles;
  pass->tclOutputPath      = std::move(tclOutput);
  pass->balanceThreshold   = balanceThreshold;
  pass->enableCommunicationOpt = enableCommunicationOpt;
  pass->singleSlrUtilThreshold = singleSlrUtilThreshold;
  return pass;
}

// Backward-compatible no-arg version.
std::unique_ptr<Pass> createLayoutPartitionPass() {
  return std::make_unique<LayoutPartitionPass>();
}

} // namespace scalehls
} // namespace mlir
