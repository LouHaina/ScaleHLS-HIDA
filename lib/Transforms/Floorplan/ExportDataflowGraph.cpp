//===- ExportDataflowGraph.cpp - Export the dataflow graph for floorplanning -===//
//
// Part of the floorplan-aware HIDA flow (docs/FloorplanAwareHIDA-Design.md).
//
// Runs after ConvertDataflowToFunc and QoREstimation. It recovers the real
// dataflow graph from the function-level IR: two calls that share a buffer or
// stream operand are connected by an edge, the earlier call in program order
// being the producer. Nodes are the callees of the top function ("units") and
// everything they call ("leaves"). The graph is written as JSON for the Python
// floorplanner, and every buffer/stream op gets a stable "floorplan_id"
// attribute so a later ApplyFloorplan pass can find it again. Units are marked
// with "floorplan_unit" and "inline_off" so that they survive as RTL
// hierarchy and can be matched by name in Vivado.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "scalehls/Dialect/HLS/HLS.h"
#include "scalehls/Dialect/HLS/Utils.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <deque>

using namespace mlir;
using namespace scalehls;
using namespace hls;

namespace {

struct NodeRecord {
  std::string name;
  std::string parent; // unit name, or "" for a unit
  bool isUnit = false;
  bool isInline = false;
  bool hasDataflow = false;
  int64_t lut = 0, dsp = 0, bram = 0;
  int64_t latency = 0, interval = 0;
  SmallVector<std::string> callees;
};

struct EdgeRecord {
  std::string src, dst, kind, channelId, scope;
  int64_t width = 0, depth = 1;
};

/// Bit width of one element of a memref (vectors count all lanes).
static int64_t getElementBits(MemRefType type) {
  auto elem = type.getElementType();
  if (auto vec = elem.dyn_cast<VectorType>())
    return vec.getNumElements() * vec.getElementTypeBitWidth();
  if (elem.isIntOrFloat())
    return elem.getIntOrFloatBitWidth();
  return 32;
}

/// Follow view-like ops to the underlying buffer value.
static Value traceBuffer(Value v) {
  while (auto view = v.getDefiningOp<ViewLikeOpInterface>())
    v = view.getViewSource();
  return v;
}

struct ExportDataflowGraph
    : public ExportDataflowGraphBase<ExportDataflowGraph> {
  ExportDataflowGraph() = default;
  ExportDataflowGraph(std::string outputPath, std::string top) {
    output = outputPath;
    topFunc = top;
  }

  llvm::MapVector<StringRef, NodeRecord> nodes;
  SmallVector<EdgeRecord> edges;
  unsigned channelCounter = 0;
  /// AXI port element (memref in the top) -> runtime buffer it is packed from.
  DenseMap<Value, Value> portToRuntimeBuffer;
  /// Runtime buffer -> ordinal used as channel id.
  DenseMap<Value, unsigned> runtimeBufferIds;
  static constexpr const char *memNode = "__mem__";

  /// Two calls in the top that read/write the same DRAM buffer through
  /// different AXI ports are still producer/consumer: map each port back to
  /// the runtime buffer packed into it by the runtime "main" function.
  void resolveAxiPorts(ModuleOp module, func::FuncOp top) {
    func::CallOp topCall;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (!hasRuntimeAttr(func))
        continue;
      func.walk([&](func::CallOp call) {
        if (call.getCallee() == top.getName())
          topCall = call;
      });
    }
    if (!topCall)
      return;
    top.walk([&](AxiPortOp port) {
      auto arg = port.getAxi().dyn_cast<BlockArgument>();
      if (!arg || arg.getOwner()->getParentOp() != top)
        return;
      if (arg.getArgNumber() >= topCall.getNumOperands())
        return;
      auto pack = topCall.getOperand(arg.getArgNumber()).getDefiningOp<AxiPackOp>();
      if (!pack)
        return;
      auto buffer = traceBuffer(pack.getElement());
      portToRuntimeBuffer[port.getElement()] = buffer;
      runtimeBufferIds.try_emplace(buffer, runtimeBufferIds.size());
    });
  }

  /// Give a buffer/stream op a stable id attribute (function:ordinal).
  std::string getOrCreateChannelId(Value v, func::FuncOp scope) {
    auto op = v.getDefiningOp();
    if (!op)
      return "";
    if (auto attr = op->getAttrOfType<StringAttr>("floorplan_id"))
      return attr.getValue().str();
    auto id = scope.getName().str() + ":" + std::to_string(channelCounter++);
    op->setAttr("floorplan_id", StringAttr::get(op->getContext(), id));
    return id;
  }

  /// Classify a call operand and compute its wire width.
  bool classify(Value v, func::FuncOp scope, bool scopeIsTop, EdgeRecord &e) {
    if (auto streamType = v.getType().dyn_cast<StreamType>()) {
      auto elemBits = streamType.getElementType().isIntOrFloat()
                          ? streamType.getElementType().getIntOrFloatBitWidth()
                          : 32;
      e.kind = elemBits == 1 ? "token" : "stream";
      e.width = elemBits + 2; // data + valid/ready
      e.depth = streamType.getDepth();
      if (auto streamOp = v.getDefiningOp<StreamOp>()) {
        e.depth = streamOp.getDepth();
        e.channelId = getOrCreateChannelId(v, scope);
      }
      return true;
    }
    auto memrefType = v.getType().dyn_cast<MemRefType>();
    if (!memrefType)
      return false; // scalars/indices are parameters, not dataflow edges
    auto base = traceBuffer(v);
    int64_t partitions = getPartitionFactors(memrefType);
    e.width = (getElementBits(memrefType) + 20) * std::max<int64_t>(partitions, 1);
    e.depth = getBufferDepth(v);
    if (base.isa<BlockArgument>()) {
      e.kind = scopeIsTop ? "port" : (isDram(memrefType) ? "dram" : "buffer");
    } else if (base.getDefiningOp<AxiPortOp>()) {
      e.kind = "axi";
    } else if (base.getDefiningOp<BufferLikeInterface>()) {
      e.kind = isDram(memrefType) ? "dram" : "buffer";
      e.channelId = getOrCreateChannelId(base, scope);
    } else {
      e.kind = "other";
    }
    return true;
  }

  /// Collect edges between the calls directly inside "func".
  void collectEdges(func::FuncOp func, bool isTop) {
    SmallVector<func::CallOp> calls;
    func.walk([&](func::CallOp call) { calls.push_back(call); });
    llvm::MapVector<Value, SmallVector<func::CallOp>> users;
    llvm::MapVector<Value, Value> keyToExample;
    for (auto call : calls)
      for (auto operand : call.getOperands()) {
        Value key = operand;
        if (isTop) {
          auto it = portToRuntimeBuffer.find(traceBuffer(operand));
          if (it != portToRuntimeBuffer.end())
            key = it->second;
          // Every AXI port is a wire to the memory subsystem: edge to __mem__.
          if (traceBuffer(operand).getDefiningOp<AxiPortOp>()) {
            EdgeRecord mem;
            mem.scope = func.getName().str();
            mem.src = call.getCallee().str();
            mem.dst = memNode;
            mem.kind = "axi";
            auto memrefType = operand.getType().dyn_cast<MemRefType>();
            mem.width = (memrefType ? getElementBits(memrefType) : 32) + 64;
            edges.push_back(mem);
          }
        }
        keyToExample.insert({key, operand});
        users[key].push_back(call);
      }
    for (auto &[key, callList] : users) {
      if (callList.size() < 2)
        continue;
      EdgeRecord proto;
      proto.scope = func.getName().str();
      Value value = keyToExample.lookup(key);
      if (!classify(value, func, isTop, proto))
        continue;
      if (key != value) { // shared runtime DRAM buffer behind AXI ports
        proto.kind = "dram";
        proto.channelId = "mem:" + std::to_string(runtimeBufferIds.lookup(key));
      }
      auto producer = callList.front();
      llvm::StringSet<> seen;
      for (auto consumer : llvm::drop_begin(callList)) {
        if (consumer.getCallee() == producer.getCallee())
          continue;
        if (!seen.insert(consumer.getCallee()).second)
          continue;
        EdgeRecord e = proto;
        e.src = producer.getCallee().str();
        e.dst = consumer.getCallee().str();
        edges.push_back(e);
      }
    }
  }

  void recordNode(func::FuncOp func, StringRef parent, bool isUnit) {
    auto &rec = nodes[func.getName()];
    rec.name = func.getName().str();
    rec.parent = parent.str();
    rec.isUnit = isUnit;
    rec.isInline = func->hasAttr("inline");
    if (auto directive = getFuncDirective(func))
      rec.hasDataflow = directive.getDataflow();
    if (auto res = getResource(func)) {
      rec.lut = res.getLut();
      rec.dsp = res.getDsp();
      rec.bram = res.getBram();
    }
    if (auto timing = getTiming(func)) {
      rec.latency = timing.getLatency();
      rec.interval = timing.getInterval();
    }
    llvm::SetVector<StringRef> callees;
    func.walk([&](func::CallOp call) { callees.insert(call.getCallee()); });
    for (auto c : callees)
      rec.callees.push_back(c.str());
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto top = getTopFunc(module, topFunc);
    if (!top) {
      module.emitError("fail to find the top function for graph export");
      return signalPassFailure();
    }
    auto context = module.getContext();

    resolveAxiPorts(module, top);

    // Units: callees of the top function in program order.
    llvm::SetVector<StringRef> unitNames;
    top.walk([&](func::CallOp call) { unitNames.insert(call.getCallee()); });

    for (auto unitName : unitNames) {
      auto unit = module.lookupSymbol<func::FuncOp>(unitName);
      if (!unit)
        continue;
      recordNode(unit, "", /*isUnit=*/true);
      unit->setAttr("floorplan_unit", UnitAttr::get(context));
      unit->setAttr("inline_off", UnitAttr::get(context));
      unit->removeAttr("inline");

      // Everything reachable from the unit belongs to it.
      std::deque<func::FuncOp> queue{unit};
      llvm::StringSet<> visited;
      visited.insert(unitName);
      while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        collectEdges(current, /*isTop=*/false);
        current.walk([&](func::CallOp call) {
          if (!visited.insert(call.getCallee()).second)
            return;
          if (auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee())) {
            recordNode(callee, unitName, /*isUnit=*/false);
            queue.push_back(callee);
          }
        });
      }
    }
    collectEdges(top, /*isTop=*/true);
    if (llvm::any_of(edges, [](const EdgeRecord &e) { return e.dst == memNode; })) {
      auto &mem = nodes[memNode];
      mem.name = memNode;
      mem.isUnit = true;
    }

    std::error_code ec;
    llvm::raw_fd_ostream os(output, ec);
    if (ec) {
      module.emitError("cannot open graph output file: ") << ec.message();
      return signalPassFailure();
    }
    llvm::json::OStream J(os, 2);
    J.object([&] {
      J.attribute("top", top.getName());
      J.attribute("resource_note",
                  "lut is not estimated by HIDA (0); dsp of a unit already "
                  "includes its callees; bram is per function");
      J.attributeArray("nodes", [&] {
        for (auto &[name, rec] : nodes)
          J.object([&] {
            J.attribute("name", rec.name);
            J.attribute("parent", rec.parent);
            J.attribute("is_unit", rec.isUnit);
            J.attribute("inline", rec.isInline);
            J.attribute("dataflow", rec.hasDataflow);
            J.attribute("lut", rec.lut);
            J.attribute("dsp", rec.dsp);
            J.attribute("bram", rec.bram);
            J.attribute("latency", rec.latency);
            J.attribute("interval", rec.interval);
            J.attributeArray("callees", [&] {
              for (auto &c : rec.callees)
                J.value(c);
            });
          });
      });
      J.attributeArray("edges", [&] {
        for (auto &e : edges)
          J.object([&] {
            J.attribute("src", e.src);
            J.attribute("dst", e.dst);
            J.attribute("kind", e.kind);
            J.attribute("width", e.width);
            J.attribute("depth", e.depth);
            J.attribute("scope", e.scope);
            J.attribute("channel", e.channelId);
          });
      });
    });
    os << "\n";
  }
};
} // namespace

std::unique_ptr<Pass>
scalehls::createExportDataflowGraphPass(std::string output,
                                        std::string topFunc) {
  return std::make_unique<ExportDataflowGraph>(output, topFunc);
}
