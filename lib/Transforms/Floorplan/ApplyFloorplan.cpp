//===- ApplyFloorplan.cpp - Apply an SLR assignment to the IR -------------===//
//
// Reads the assignment produced by tools/floorplan/floorplan.py and annotates
// the IR:
//   * each function listed in "assignment" gets an "slr" attribute;
//   * each channel listed in "channels" (keyed by the "floorplan_id" set by
//     ExportDataflowGraph) gets a "fifo_depth" attribute on its stream op, which
//     EmitHLSCpp turns into "#pragma HLS stream depth=N";
//   * functions listed in "units" are marked "floorplan_unit" and "inline_off".
//
// Converting cut on-chip buffers into streams is left to a later pass; such
// channels are reported with a warning so the floorplanner can keep them
// grouped in the meantime.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "scalehls/Dialect/HLS/HLS.h"
#include "scalehls/Transforms/Passes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;
using namespace scalehls;
using namespace hls;

namespace {
struct ApplyFloorplan : public ApplyFloorplanBase<ApplyFloorplan> {
  ApplyFloorplan() = default;
  ApplyFloorplan(std::string assignmentPath) { assignment = assignmentPath; }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    auto file = llvm::MemoryBuffer::getFile(assignment);
    if (!file) {
      module.emitError("cannot read floorplan assignment file: ") << assignment;
      return signalPassFailure();
    }
    auto parsed = llvm::json::parse(file.get()->getBuffer());
    if (!parsed) {
      module.emitError("cannot parse floorplan assignment json");
      llvm::consumeError(parsed.takeError());
      return signalPassFailure();
    }
    auto *root = parsed->getAsObject();
    if (!root) {
      module.emitError("floorplan assignment must be a json object");
      return signalPassFailure();
    }

    unsigned numAssigned = 0;
    if (auto *assign = root->getObject("assignment")) {
      for (auto &kv : *assign) {
        auto func = module.lookupSymbol<func::FuncOp>(kv.first);
        if (!func) {
          module.emitWarning("floorplan assignment names unknown function ")
              << kv.first;
          continue;
        }
        if (auto slr = kv.second.getAsInteger()) {
          func->setAttr("slr", IntegerAttr::get(IntegerType::get(context, 32),
                                                *slr));
          ++numAssigned;
        }
      }
    }

    if (auto *units = root->getArray("units")) {
      for (auto &u : *units) {
        auto name = u.getAsString();
        if (!name)
          continue;
        if (auto func = module.lookupSymbol<func::FuncOp>(*name)) {
          func->setAttr("floorplan_unit", UnitAttr::get(context));
          func->setAttr("inline_off", UnitAttr::get(context));
          func->removeAttr("inline");
        }
      }
    }

    // Channels keyed by floorplan_id.
    llvm::StringMap<Operation *> channels;
    module.walk([&](Operation *op) {
      if (auto id = op->getAttrOfType<StringAttr>("floorplan_id"))
        channels[id.getValue()] = op;
    });
    unsigned numChannels = 0;
    if (auto *chan = root->getObject("channels")) {
      for (auto &kv : *chan) {
        auto it = channels.find(kv.first);
        if (it == channels.end()) {
          module.emitWarning("floorplan channel id not found in IR: ")
              << kv.first;
          continue;
        }
        auto *obj = kv.second.getAsObject();
        if (!obj)
          continue;
        auto depth = obj->getInteger("depth");
        if (!depth)
          continue;
        if (auto stream = dyn_cast<StreamOp>(it->second)) {
          stream->setAttr("fifo_depth",
                          IntegerAttr::get(IntegerType::get(context, 32),
                                           *depth));
          ++numChannels;
        } else {
          it->second->emitWarning(
              "cut on-chip buffer cannot be pipelined yet; keep producer and "
              "consumer in the same SLR or convert it to a stream");
        }
      }
    }
    llvm::errs() << "=== APPLY FLOORPLAN: " << numAssigned
                 << " functions assigned, " << numChannels
                 << " stream depths set\n";
  }
};
} // namespace

std::unique_ptr<Pass>
scalehls::createApplyFloorplanPass(std::string assignment) {
  return std::make_unique<ApplyFloorplan>(assignment);
}
