//===- PhysicalFloorplan.cpp - Physical-feedback floorplanning ---*- C++ -*-===//
//
// Iterative, physically-informed floorplanning of HIDA dataflow designs
// (PR #4 design doc, Sec. 8: compile -> implement -> feed back -> recompile).
//
// The pass places the physical blocks of the top function on slots of the
// device (SLRs, or half-SLR bands) and emits a Vivado pblock script:
//
//   * nodes   : every dataflow node called by the top function (func.call),
//               every AXI bundle (hls.axi.bundle "axi_N" -> m_axi adapter
//               axi_N_m_axi_U), the top-level controller ("top"), and any
//               additional netlist cell reported by the feedback file
//               (s_axilite control, top-level buffers);
//   * sizes   : measured per-cell utilization from the feedback file, or, in
//               iteration 0, a proxy from the QoR estimation attributes;
//   * edges   : measured top-level connectivity (bits) from the feedback file,
//               or, in iteration 0, the IR structure (node <-> its AXI bundles,
//               node <-> top-level buffers it shares with other nodes,
//               node <-> controller);
//   * critical: unit pairs on the worst paths of the previous implementation
//               (extra weight, PR #4 Sec. 8);
//   * stability: moving a node away from its slot of the previous iteration
//               costs a small amount, so the loop converges instead of
//               reshuffling equal-cost solutions.
//
// Objective (same as tools/floorplan/grid_floorplan.py, solved there with
// CP-SAT; here with a deterministic move/swap local search):
//   sum_e 10 w_e (|dx| + |dy| + slrPen |dslr|) + balance * maxUtil
//   + sum_n stay * share(n) * [slot(n) != prev(n)]
//   subject to  load(slot, r) <= rho * capacity(slot, r).
//
// From the path classes of the previous run the pass also chooses the next
// action of the loop (written to the output JSON): an RTL pipelining pass
// ("ctrl", "ungate_axi"), a refinement of the slot grid (slr -> band), more
// feedback, or stop.
//
//===----------------------------------------------------------------------===//

#include "scalehls/Dialect/HLS/HLS.h"
#include "scalehls/Dialect/HLS/Utils.h"
#include "scalehls/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <random>
#include <set>

using namespace mlir;
using namespace scalehls;
using namespace hls;

namespace {

constexpr int kNumRes = 5; // lut, ff, dsp, bram, uram
const char *kResName[kNumRes] = {"lut", "ff", "dsp", "bram", "uram"};

struct FpNode {
  std::string key;         // stable id: callee name, bundle name, "top", cell
  std::string cell;        // netlist cell name if known ("" -> pattern)
  std::string kind;        // node | axi | top | cell
  double res[kNumRes] = {0, 0, 0, 0, 0};
  SmallVector<Operation *, 2> ops; // IR ops to annotate
};

struct FpSlot {
  std::string name;
  int x = 0, y = 0, slr = 0;
  double cap[kNumRes] = {0, 0, 0, 0, 0};
  std::string range; // CLOCKREGION_.. range for the pblock
};

struct FpEdge {
  int a, b;
  double w;
};

/// Clock region table (tools/floorplan/vivado/extract_graph.tcl output):
/// cr,x,y,slice,dsp,ramb36,uram
struct ClockRegion {
  int x, y;
  double res[kNumRes];
};

struct PhysicalFloorplanPass
    : public PassWrapper<PhysicalFloorplanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PhysicalFloorplanPass)

  PhysicalFloorplanPass() = default;
  PhysicalFloorplanPass(const PhysicalFloorplanPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "scalehls-physical-floorplan";
  }
  StringRef getDescription() const override {
    return "Iterative physical-feedback floorplanning of HIDA dataflow nodes";
  }

  Option<std::string> topFunc{*this, "top-func", llvm::cl::init(""),
                              llvm::cl::desc("Top function (default: the "
                                             "function marked top_func)")};
  Option<std::string> deviceFile{
      *this, "device", llvm::cl::init(""),
      llvm::cl::desc("Clock-region table (csv: cr,x,y,slice,dsp,ramb36,uram)")};
  Option<std::string> feedbackFile{
      *this, "feedback", llvm::cl::init(""),
      llvm::cl::desc("Feedback JSON of the previous implementation "
                     "(tools/floorplan/make_feedback.py); empty: iteration 0")};
  Option<std::string> outTcl{*this, "out-tcl",
                             llvm::cl::init("floorplan.tcl"),
                             llvm::cl::desc("Vivado pblock script")};
  Option<std::string> outJson{*this, "out-json",
                              llvm::cl::init("floorplan.json"),
                              llvm::cl::desc("Assignment and next action")};
  Option<std::string> slotMode{
      *this, "slots", llvm::cl::init("auto"),
      llvm::cl::desc("slr | band | auto (slr, refined to band by feedback)")};
  Option<int> bandRows{*this, "band-rows", llvm::cl::init(2),
                       llvm::cl::desc("Clock-region rows per band")};
  Option<int> rowsPerSlr{*this, "rows-per-slr", llvm::cl::init(4),
                         llvm::cl::desc("Clock-region rows per SLR")};
  Option<std::string> strip{
      *this, "strip", llvm::cl::init(""),
      llvm::cl::desc("Clock-region columns x0-x1 the design is confined to")};
  Option<double> rho{*this, "rho", llvm::cl::init(0.75),
                     llvm::cl::desc("Max utilization per slot")};
  Option<double> balance{*this, "balance", llvm::cl::init(2000),
                         llvm::cl::desc("Weight of max slot utilization")};
  Option<int> slrPen{*this, "slr-pen", llvm::cl::init(20),
                     llvm::cl::desc("Extra distance per SLR boundary")};
  Option<double> stay{*this, "stay", llvm::cl::init(1000),
                      llvm::cl::desc("Weight of the stability term")};
  Option<double> critWeight{*this, "critical-weight", llvm::cl::init(50),
                            llvm::cl::desc("Weight of critical-path pairs")};
  Option<std::string> pinNodes{
      *this, "pin", llvm::cl::init("top,control_s_axi_U"),
      llvm::cl::desc("Nodes pinned to the middle slot")};
  Option<double> derate{*this, "density-derate", llvm::cl::init(0.85),
                        llvm::cl::desc("Capacity factor applied to a slot whose units "
                                       "show wire-dominated internal paths")};
  Option<double> minRho{*this, "min-rho", llvm::cl::init(0.45),
                        llvm::cl::desc("Lower bound of a derated slot capacity")};
  Option<int> restarts{*this, "restarts", llvm::cl::init(16),
                       llvm::cl::desc("Local-search restarts")};
  Option<int> perturbIters{*this, "perturb", llvm::cl::init(400),
                           llvm::cl::desc("Iterated-local-search rounds")};

  // ---------------------------------------------------------------- state
  std::vector<FpNode> nodes;
  llvm::StringMap<int> index;
  std::vector<FpSlot> slots;
  std::vector<FpEdge> edges;
  std::map<std::pair<int, int>, double> edgeMap;
  std::map<std::string, std::string> prevAssign;
  std::map<std::string, double> pathClasses;
  std::vector<std::string> appliedRtl;
  std::string prevSlotMode;
  double prevWns = NAN;
  bool haveFeedback = false;
  std::vector<std::string> dropped;
  std::map<std::string, double> slotRhoIn;  // capacity limits of the previous iteration
  std::map<std::string, double> denseUnits; // cell -> wire-dominated internal paths
  std::vector<double> rhoS;                 // capacity limit per slot

  int nodeFor(const std::string &key, const std::string &kind) {
    auto it = index.find(key);
    if (it != index.end())
      return it->second;
    FpNode n;
    n.key = key;
    n.kind = kind;
    nodes.push_back(n);
    index[key] = nodes.size() - 1;
    return nodes.size() - 1;
  }

  /// Netlist cell name -> stable node key.
  std::string keyOfCell(StringRef cell) {
    static llvm::Regex unit("^grp_(.+)_fu_[0-9]+$");
    static llvm::Regex unitU0("^(.+)_U0$");
    static llvm::Regex axi("^(axi_[0-9]+)_m_axi_U$");
    SmallVector<StringRef, 2> m;
    if (unit.match(cell, &m) && index.count(m[1]))
      return m[1].str();
    if (unitU0.match(cell, &m) && index.count(m[1]))
      return m[1].str();
    if (axi.match(cell, &m) && index.count(m[1]))
      return m[1].str();
    if (cell == "(forward)" || cell == "top")
      return "top";
    return cell.str();
  }

  void addEdge(int a, int b, double w) {
    if (a == b || w <= 0)
      return;
    edgeMap[{std::min(a, b), std::max(a, b)}] += w;
  }
  /// ILP edge weight: round(10 * w).
  static double ilpWeight(double w) { return std::round(10.0 * w); }

  // ------------------------------------------------------------ IR graph
  LogicalResult buildFromIR(func::FuncOp top) {
    int t = nodeFor("top", "top");
    nodes[t].ops.push_back(top);
    // AXI bundles.
    DenseMap<Value, int> bundleOf; // port value -> bundle node
    top.walk([&](AxiBundleOp b) {
      int n = nodeFor(b.getName().str(), "axi");
      nodes[n].ops.push_back(b);
    });
    top.walk([&](AxiPortOp p) {
      bundleOf[p.getResult()] = index[p.getBundleName().str()];
    });
    // Dataflow nodes and their structural edges.
    DenseMap<Value, SmallVector<int, 4>> sharedBuf;
    top.walk([&](func::CallOp call) {
      int n = nodeFor(call.getCallee().str(), "node");
      nodes[n].ops.push_back(call);
      if (auto res = getResource(call)) {
        // Proxy until measured sizes are fed back (QoR estimation reports
        // DSP/BRAM; its LUT field is not populated for HIDA designs).
        nodes[n].res[0] = std::max<double>(res.getLut(),
                                           150.0 * res.getDsp() + 2000);
        nodes[n].res[1] = nodes[n].res[0];
        nodes[n].res[2] = res.getDsp();
        nodes[n].res[3] = 0; // QoR BRAM is far above the implemented value
      }
      addEdge(n, t, 64); // ap_start/done/continue + scalar arguments
      for (Value arg : call.getOperands()) {
        auto it = bundleOf.find(arg);
        if (it != bundleOf.end())
          addEdge(n, it->second, 256); // AXI handshake + data + address
        else if (arg.getType().isa<MemRefType>())
          sharedBuf[arg].push_back(n);
      }
    });
    for (auto &kv : sharedBuf)
      for (unsigned i = 0; i < kv.second.size(); ++i)
        for (unsigned j = i + 1; j < kv.second.size(); ++j)
          addEdge(kv.second[i], kv.second[j], 64);
    for (auto &n : nodes)
      if (n.kind == "axi" && n.res[0] == 0) {
        n.res[0] = 1500; // typical Vitis m_axi adapter
        n.res[1] = 2500;
        n.res[3] = 1;
      }
    return success();
  }

  // ------------------------------------------------------------- feedback
  LogicalResult readFeedback() {
    auto buf = llvm::MemoryBuffer::getFile(feedbackFile);
    if (!buf)
      return emitError(getOperation().getLoc(),
                       "cannot read feedback " + feedbackFile);
    auto parsed = llvm::json::parse((*buf)->getBuffer());
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return emitError(getOperation().getLoc(), "malformed feedback JSON");
    }
    auto *root = parsed->getAsObject();
    haveFeedback = true;
    // Measured cells replace the proxies; unknown cells become nodes.
    if (auto *cells = root->getObject("cells")) {
      for (auto &kv : *cells) {
        std::string key = keyOfCell(kv.first);
        int n = nodeFor(key, index.count(key) ? nodes[index[key]].kind : "cell");
        if (nodes[n].kind != "top")
          nodes[n].cell = kv.first.str();
        if (auto *o = kv.second.getAsObject())
          for (int r = 0; r < kNumRes; ++r)
            if (auto v = o->getNumber(kResName[r]))
              nodes[n].res[r] = *v;
      }
      edgeMap.clear(); // measured connectivity replaces the IR estimate
      // IR nodes without a netlist cell were optimized away by HLS/synthesis
      // (e.g. an unused AXI bundle): drop them, keep the IR ops unannotated.
      std::vector<FpNode> kept;
      for (auto &n : nodes)
        if (n.kind == "top" || n.kind == "cell" || !n.cell.empty())
          kept.push_back(n);
        else
          dropped.push_back(n.key);
      nodes = std::move(kept);
      index.clear();
      for (unsigned i = 0; i < nodes.size(); ++i)
        index[nodes[i].key] = i;
    }
    if (auto *es = root->getArray("edges"))
      for (auto &e : *es)
        if (auto *o = e.getAsObject()) {
          auto a = o->getString("a"), b = o->getString("b");
          auto w = o->getNumber("bits");
          if (a && b && w)
            addEdge(nodeFor(keyOfCell(*a), "cell"),
                    nodeFor(keyOfCell(*b), "cell"), *w);
        }
    if (auto *cs = root->getArray("critical"))
      for (auto &e : *cs)
        if (auto *o = e.getAsObject()) {
          auto a = o->getString("a"), b = o->getString("b");
          auto w = o->getNumber("weight");
          if (a && b && w)
            addEdge(nodeFor(keyOfCell(*a), "cell"),
                    nodeFor(keyOfCell(*b), "cell"), critWeight * *w);
        }
    if (auto *prev = root->getObject("previous"))
      for (auto &kv : *prev)
        if (auto s = kv.second.getAsString())
          prevAssign[kv.first.str()] = s->str();
    if (auto *pc = root->getObject("path_classes"))
      for (auto &kv : *pc)
        if (auto v = kv.second.getAsNumber())
          pathClasses[kv.first.str()] = *v;
    if (auto *ar = root->getArray("applied_rtl"))
      for (auto &v : *ar)
        if (auto s = v.getAsString())
          appliedRtl.push_back(s->str());
    if (auto *sr = root->getObject("slot_rho"))
      for (auto &kv : *sr)
        if (auto v = kv.second.getAsNumber())
          slotRhoIn[kv.first.str()] = *v;
    if (auto *du = root->getObject("dense_units"))
      for (auto &kv : *du)
        if (auto v = kv.second.getAsNumber())
          denseUnits[kv.first.str()] = *v;
    if (auto s = root->getString("slot_mode"))
      prevSlotMode = s->str();
    if (auto w = root->getNumber("wns"))
      prevWns = *w;
    return success();
  }

  // --------------------------------------------------------------- device
  LogicalResult readDevice(std::vector<ClockRegion> &crs) {
    auto buf = llvm::MemoryBuffer::getFile(deviceFile);
    if (!buf)
      return emitError(getOperation().getLoc(),
                       "cannot read device table " + deviceFile);
    SmallVector<StringRef, 128> lines;
    (*buf)->getBuffer().split(lines, '\n', -1, false);
    bool anyBram = false;
    for (unsigned i = 1; i < lines.size(); ++i) {
      SmallVector<StringRef, 8> f;
      lines[i].trim().split(f, ',');
      if (f.size() < 7)
        continue;
      ClockRegion c;
      long v[6];
      for (int k = 0; k < 6; ++k)
        if (f[k + 1].getAsInteger(10, v[k]))
          v[k] = 0;
      c.x = v[0];
      c.y = v[1];
      c.res[0] = 8.0 * v[2];
      c.res[1] = 16.0 * v[2];
      c.res[2] = v[3];
      c.res[3] = v[4];
      c.res[4] = v[5];
      anyBram |= v[4] > 0;
      crs.push_back(c);
    }
    if (!anyBram) // site query found no RAMB36: device average
      for (auto &c : crs)
        c.res[3] = 21;
    if (!strip.empty()) {
      int x0, x1;
      if (sscanf(strip.c_str(), "%d-%d", &x0, &x1) != 2)
        return emitError(getOperation().getLoc(), "bad strip " + strip);
      crs.erase(std::remove_if(crs.begin(), crs.end(),
                               [&](const ClockRegion &c) {
                                 return c.x < x0 || c.x > x1;
                               }),
                crs.end());
    }
    if (crs.empty())
      return emitError(getOperation().getLoc(), "empty device table");
    return success();
  }

  void buildSlots(const std::vector<ClockRegion> &crs, StringRef mode) {
    int xmin = 1 << 30, xmax = -1, ymax = 0;
    for (auto &c : crs) {
      xmin = std::min(xmin, c.x);
      xmax = std::max(xmax, c.x);
      ymax = std::max(ymax, c.y);
    }
    int rows = mode == "band" ? bandRows.getValue() : rowsPerSlr.getValue();
    std::map<int, FpSlot> byRow;
    for (auto &c : crs) {
      int y0 = c.y / rows * rows;
      auto &s = byRow[y0];
      s.y = y0;
      s.slr = y0 / rowsPerSlr;
      for (int r = 0; r < kNumRes; ++r)
        s.cap[r] += c.res[r];
    }
    for (auto &kv : byRow) {
      FpSlot s = kv.second;
      s.name = (mode == "band" ? "B" : "SLR") +
               std::to_string(mode == "band" ? s.y : s.slr);
      s.range = "CLOCKREGION_X" + std::to_string(xmin) + "Y" +
                std::to_string(s.y) + ":CLOCKREGION_X" + std::to_string(xmax) +
                "Y" + std::to_string(std::min(s.y + rows - 1, ymax));
      slots.push_back(s);
    }
  }

  // --------------------------------------------------------------- solver
  struct Solution {
    std::vector<int> slot;
    double cost = INFINITY;
  };

  std::vector<std::vector<std::pair<int, double>>> adj;
  std::vector<int> pinned; // -1 = free
  std::vector<int> prevSlot;
  std::vector<double> stayCost;
  std::vector<std::vector<int>> followers; // leader -> nodes moved with it

  double edgeCost(int a, int sa, int b, int sb) const {
    const FpSlot &A = slots[sa], &B = slots[sb];
    double d = std::abs(A.x - B.x) + std::abs(A.y - B.y) +
               slrPen * std::abs(A.slr - B.slr);
    return d;
  }

  double maxUtil(const std::vector<std::array<double, kNumRes>> &load) const {
    double m = 0;
    for (unsigned s = 0; s < slots.size(); ++s)
      for (int r : {0, 2})
        if (slots[s].cap[r] > 0)
          m = std::max(m, load[s][r] / slots[s].cap[r]);
    return m;
  }

  double balTerm(const std::vector<std::array<double, kNumRes>> &load) const {
    return balance * 1000.0 * maxUtil(load);
  }

  bool fits(const std::vector<std::array<double, kNumRes>> &load, int s,
            int n) const {
    for (int r = 0; r < kNumRes; ++r)
      if (nodes[n].res[r] > 0 &&
          load[s][r] + nodes[n].res[r] > rhoS[s] * slots[s].cap[r] + 1e-9)
        return false;
    return true;
  }

  double totalCost(const std::vector<int> &slot,
                   const std::vector<std::array<double, kNumRes>> &load) const {
    double c = 0;
    for (auto &e : edges)
      c += e.w * edgeCost(e.a, slot[e.a], e.b, slot[e.b]);
    c += balTerm(load);
    for (unsigned n = 0; n < nodes.size(); ++n)
      if (prevSlot[n] >= 0 && slot[n] != prevSlot[n])
        c += stayCost[n];
    return c;
  }

  Solution localSearch(std::vector<int> slot, std::mt19937 &rng) {
    unsigned N = nodes.size(), S = slots.size();
    std::vector<std::array<double, kNumRes>> load(S);
    for (auto &l : load)
      l.fill(0);
    for (unsigned n = 0; n < N; ++n)
      for (int r = 0; r < kNumRes; ++r)
        load[slot[n]][r] += nodes[n].res[r];
    auto nodeEdgeCost = [&](int n, int s) {
      double c = 0;
      for (auto &[m, w] : adj[n])
        c += w * edgeCost(n, s, m, slot[m]);
      if (prevSlot[n] >= 0 && s != prevSlot[n])
        c += stayCost[n];
      return c;
    };
    std::vector<int> order(N);
    for (unsigned i = 0; i < N; ++i)
      order[i] = i;
    for (int pass = 0; pass < 200; ++pass) {
      bool improved = false;
      std::shuffle(order.begin(), order.end(), rng);
      for (int n : order) {
        if (pinned[n] >= 0)
          continue;
        int s0 = slot[n];
        double base = nodeEdgeCost(n, s0);
        double mu0 = balTerm(load);
        int bestS = -1, bestM = -1;
        double bestD = -1e-9;
        for (unsigned s = 0; s < S; ++s) {
          if ((int)s == s0)
            continue;
          // single move
          for (int r = 0; r < kNumRes; ++r)
            load[s0][r] -= nodes[n].res[r];
          if (fits(load, s, n)) {
            for (int r = 0; r < kNumRes; ++r)
              load[s][r] += nodes[n].res[r];
            double d = nodeEdgeCost(n, s) - base + (balTerm(load) - mu0);
            for (int r = 0; r < kNumRes; ++r)
              load[s][r] -= nodes[n].res[r];
            if (d < bestD) {
              bestD = d;
              bestS = s;
              bestM = -1;
            }
          }
          for (int r = 0; r < kNumRes; ++r)
            load[s0][r] += nodes[n].res[r];
          // swap with a node of slot s
          for (unsigned m = 0; m < N; ++m) {
            if (slot[m] != (int)s || pinned[m] >= 0)
              continue;
            for (int r = 0; r < kNumRes; ++r) {
              load[s0][r] += nodes[m].res[r] - nodes[n].res[r];
              load[s][r] += nodes[n].res[r] - nodes[m].res[r];
            }
            bool ok = true;
            for (int r = 0; r < kNumRes && ok; ++r)
              ok = load[s0][r] <= rhoS[s0] * slots[s0].cap[r] + 1e-9 &&
                   load[s][r] <= rhoS[s] * slots[s].cap[r] + 1e-9;
            if (ok) {
              double before = base + nodeEdgeCost(m, s);
              slot[n] = s;
              slot[m] = s0;
              double after = nodeEdgeCost(n, s) + nodeEdgeCost(m, s0);
              // an edge n-m is counted twice in both terms: no correction
              slot[n] = s0;
              slot[m] = s;
              double d = after - before + (balTerm(load) - mu0);
              if (d < bestD) {
                bestD = d;
                bestS = s;
                bestM = m;
              }
            }
            for (int r = 0; r < kNumRes; ++r) {
              load[s0][r] -= nodes[m].res[r] - nodes[n].res[r];
              load[s][r] -= nodes[n].res[r] - nodes[m].res[r];
            }
          }
        }
        if (bestS >= 0) {
          for (int r = 0; r < kNumRes; ++r) {
            load[s0][r] -= nodes[n].res[r];
            load[bestS][r] += nodes[n].res[r];
          }
          slot[n] = bestS;
          if (bestM >= 0) {
            for (int r = 0; r < kNumRes; ++r) {
              load[bestS][r] -= nodes[bestM].res[r];
              load[s0][r] += nodes[bestM].res[r];
            }
            slot[bestM] = s0;
          }
          improved = true;
        }
      }
      // group moves: a leader with its followers (e.g. a dataflow node with
      // the AXI adapters only it uses), so balance can be fixed without
      // stretching the leader's private connections
      for (int n : order) {
        if (followers[n].empty() || pinned[n] >= 0)
          continue;
        std::vector<int> grp{n};
        for (int m : followers[n])
          if (slot[m] == slot[n] && pinned[m] < 0)
            grp.push_back(m);
        int s0 = slot[n];
        std::set<int> inGrp(grp.begin(), grp.end());
        auto grpCost = [&](int s) {
          double c = 0;
          for (int g : grp) {
            for (auto &[m, w] : adj[g])
              if (!inGrp.count(m))
                c += w * edgeCost(g, s, m, slot[m]);
            if (prevSlot[g] >= 0 && s != prevSlot[g])
              c += stayCost[g];
          }
          return c;
        };
        double base = grpCost(s0), mu0 = balTerm(load);
        int bestS = -1;
        double bestD = -1e-9;
        for (unsigned s = 0; s < S; ++s) {
          if ((int)s == s0)
            continue;
          for (int g : grp)
            for (int r = 0; r < kNumRes; ++r) {
              load[s0][r] -= nodes[g].res[r];
              load[s][r] += nodes[g].res[r];
            }
          bool ok = true;
          for (int r = 0; r < kNumRes && ok; ++r)
            ok = load[s][r] <= rhoS[s] * slots[s].cap[r] + 1e-9;
          if (ok) {
            double d = grpCost(s) - base + (balTerm(load) - mu0);
            if (d < bestD) {
              bestD = d;
              bestS = s;
            }
          }
          for (int g : grp)
            for (int r = 0; r < kNumRes; ++r) {
              load[s0][r] += nodes[g].res[r];
              load[s][r] -= nodes[g].res[r];
            }
        }
        if (bestS >= 0) {
          for (int g : grp) {
            for (int r = 0; r < kNumRes; ++r) {
              load[s0][r] -= nodes[g].res[r];
              load[bestS][r] += nodes[g].res[r];
            }
            slot[g] = bestS;
          }
          improved = true;
        }
      }
      if (!improved)
        break;
    }
    Solution sol;
    sol.slot = slot;
    sol.cost = totalCost(slot, load);
    return sol;
  }

  /// Greedy initial assignment: biggest nodes first, each to the feasible slot
  /// with the lowest incremental cost (connections to already placed nodes and
  /// load balance).  Randomized tie-breaking across restarts.
  bool greedy(std::vector<int> &slot, std::mt19937 &rng, bool usePrev) {
    unsigned N = nodes.size(), S = slots.size();
    std::vector<std::array<double, kNumRes>> load(S);
    for (auto &l : load)
      l.fill(0);
    slot.assign(N, -1);
    std::vector<int> order(N);
    for (unsigned i = 0; i < N; ++i)
      order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      if ((pinned[a] >= 0) != (pinned[b] >= 0))
        return pinned[a] >= 0;
      return nodes[a].res[0] + 100 * nodes[a].res[2] >
             nodes[b].res[0] + 100 * nodes[b].res[2];
    });
    for (int n : order) {
      int best = -1;
      double bestC = INFINITY;
      for (unsigned s = 0; s < S; ++s) {
        if (pinned[n] >= 0 && (int)s != pinned[n])
          continue;
        if (!fits(load, s, n))
          continue;
        double c = 0;
        for (auto &[m, w] : adj[n])
          if (slot[m] >= 0)
            c += w * edgeCost(n, s, m, slot[m]);
        if (prevSlot[n] >= 0 && (int)s != prevSlot[n])
          c += usePrev ? 1e12 : stayCost[n];
        for (int r = 0; r < kNumRes; ++r)
          load[s][r] += nodes[n].res[r];
        c += balTerm(load);
        for (int r = 0; r < kNumRes; ++r)
          load[s][r] -= nodes[n].res[r];
        c += std::uniform_real_distribution<double>(0, 1e-3)(rng);
        if (c < bestC) {
          bestC = c;
          best = s;
        }
      }
      if (best < 0)
        return false;
      slot[n] = best;
      for (int r = 0; r < kNumRes; ++r)
        load[best][r] += nodes[n].res[r];
    }
    return true;
  }

  // --------------------------------------------------------------- output
  void emitTcl(const std::vector<int> &slot) {
    std::error_code ec;
    llvm::raw_fd_ostream os(outTcl, ec);
    if (ec) {
      emitError(getOperation().getLoc(), "cannot write " + outTcl);
      return signalPassFailure();
    }
    os << "# Generated by scalehls-physical-floorplan ("
       << (haveFeedback ? "with feedback" : "iteration 0") << ")\n"
       << "catch {delete_pblocks [get_pblocks]}\n";
    for (unsigned s = 0; s < slots.size(); ++s) {
      std::vector<int> members;
      for (unsigned n = 0; n < nodes.size(); ++n)
        if (slot[n] == (int)s)
          members.push_back(n);
      if (members.empty())
        continue;
      std::string pb = "pb_" + slots[s].name;
      os << "create_pblock " << pb << "\n"
         << "resize_pblock " << pb << " -add {" << slots[s].range << "}\n"
         << "set_property CONTAIN_ROUTING false [get_pblocks " << pb << "]\n"
         << "set_property SNAPPING_MODE ON [get_pblocks " << pb << "]\n"
         << "set cells [list]\n";
      for (int n : members) {
        const FpNode &N = nodes[n];
        if (N.kind == "top") {
          // add the collection directly: converting it to a Tcl list is
          // truncated by Vivado (tcl.collectionResultDisplayLimit, 500)
          os << "add_cells_to_pblock " << pb
             << " [get_cells -filter {IS_PRIMITIVE==1}]\n";
          continue;
        }
        if (!N.cell.empty())
          os << "set c [get_cells -quiet {" << N.cell << "}]\n";
        else if (N.kind == "axi")
          os << "set c [get_cells -quiet {" << N.key << "_m_axi_U}]\n";
        else
          os << "set c [get_cells -quiet -regexp {^(grp_)?" << N.key
             << "(_[0-9]+)?(_fu_[0-9]+|_U0)$}]\n";
        if (N.kind == "axi" && N.cell.empty())
          // iteration 0: an AXI bundle may have been optimized away by HLS or
          // synthesis; nothing to place then
          os << "if {[llength $c] == 0} { puts \"floorplan: " << N.key
             << "_m_axi_U not in netlist, skipped\" }\n"
             << "if {[llength $c] > 1} { error \"floorplan: " << N.key
             << " matched [llength $c] cells\" }\n";
        else
          os << "if {[llength $c] != 1} { error \"floorplan: " << N.key
             << " matched [llength $c] cells\" }\n";
        os << "lappend cells {*}$c\n";
      }
      os << "add_cells_to_pblock " << pb << " $cells\n";
    }
  }

  // ----------------------------------------------------------------- main
  void runOnOperation() override {
    ModuleOp module = getOperation();
    func::FuncOp top = getTopFunc(module, topFunc);
    if (!top) {
      emitError(module.getLoc(), "no top function");
      return signalPassFailure();
    }
    if (failed(buildFromIR(top)))
      return signalPassFailure();
    if (!feedbackFile.empty() && failed(readFeedback()))
      return signalPassFailure();

    // ---- next action of the loop, from the previous run's path classes
    std::string action = haveFeedback ? "feedback" : "initial";
    std::vector<std::string> rtl = appliedRtl;
    std::string mode = slotMode == "auto"
                           ? (prevSlotMode.empty() ? "slr" : prevSlotMode)
                           : slotMode.getValue();
    if (haveFeedback && !pathClasses.empty()) {
      auto dom = std::max_element(
          pathClasses.begin(), pathClasses.end(),
          [](auto &a, auto &b) { return a.second < b.second; });
      auto has = [&](StringRef p) {
        return std::find(rtl.begin(), rtl.end(), p.str()) != rtl.end();
      };
      if (dom->first == "ctrl_broadcast" && !has("ctrl")) {
        rtl.push_back("ctrl");
        action = "rtl:ctrl";
      } else if (dom->first == "fsm_axi_gate" && !has("ungate_axi")) {
        if (!has("ctrl"))
          rtl.push_back("ctrl");
        rtl.push_back("ungate_axi");
        action = "rtl:ungate_axi";
      } else if (dom->first == "unit_density") {
        action = "density:derate";
      } else if (dom->first == "placement_local" && slotMode == "auto" &&
                 mode == "slr") {
        mode = "band";
        action = "feedback+refine:band";
      } else if (dom->first == "unit_internal") {
        action = "stop";
      }
    }

    std::vector<ClockRegion> crs;
    if (failed(readDevice(crs)))
      return signalPassFailure();
    buildSlots(crs, mode);
    rhoS.assign(slots.size(), rho);
    for (unsigned s = 0; s < slots.size(); ++s) {
      auto it = slotRhoIn.find(slots[s].name);
      if (it != slotRhoIn.end())
        rhoS[s] = it->second;
    }
    if (action == "density:derate") {
      // slots whose units have wire-dominated internal paths are too dense:
      // lower their capacity limit; the limit carries over to later iterations
      std::set<std::string> hot;
      for (auto &kv : denseUnits) {
        auto it = prevAssign.find(keyOfCell(kv.first));
        if (it != prevAssign.end())
          hot.insert(it->second);
      }
      for (unsigned s = 0; s < slots.size(); ++s)
        if (hot.count(slots[s].name))
          rhoS[s] = std::max<double>(minRho, rhoS[s] * derate);
    }

    // ---- solver setup
    unsigned N = nodes.size();
    edges.clear();
    for (auto &kv : edgeMap)
      edges.push_back({kv.first.first, kv.first.second, ilpWeight(kv.second)});
    adj.assign(N, {});
    for (auto &e : edges) {
      adj[e.a].push_back({e.b, e.w});
      adj[e.b].push_back({e.a, e.w});
    }
    followers.assign(N, {});
    for (unsigned m = 0; m < N; ++m) {
      int lead = -1;
      double bw = 0;
      for (auto &[n, w] : adj[m])
        if (nodes[n].kind != "top" && w > bw) {
          bw = w;
          lead = n;
        }
      if (lead >= 0 && nodes[m].kind != "node" && nodes[lead].kind == "node")
        followers[lead].push_back(m);
    }
    pinned.assign(N, -1);
    {
      int mid = -1;
      for (unsigned s = 0; s < slots.size(); ++s)
        if (slots[s].slr == (slots.back().slr + slots.front().slr) / 2) {
          mid = s;
          if (mode != "band")
            break; // first slot of the middle SLR
        }
      SmallVector<StringRef, 4> pins;
      StringRef(pinNodes).split(pins, ',', -1, false);
      for (auto p : pins) {
        std::string k = keyOfCell(p);
        if (index.count(k) && mid >= 0)
          pinned[index[k]] = mid;
      }
    }
    prevSlot.assign(N, -1);
    stayCost.assign(N, 0);
    double totLut = 0;
    for (auto &n : nodes)
      totLut += std::max(n.res[0], 1.0);
    for (unsigned n = 0; n < N; ++n) {
      auto it = prevAssign.find(nodes[n].key);
      if (it == prevAssign.end())
        continue;
      for (unsigned s = 0; s < slots.size(); ++s)
        if (slots[s].name == it->second)
          prevSlot[n] = s;
      stayCost[n] =
          std::max(1.0, std::round(stay * 1000.0 *
                                   std::max(nodes[n].res[0], 1.0) / totLut));
    }

    Solution best;
    std::mt19937 rng(1);
    for (int r = 0; r < std::max(1, (int)restarts); ++r) {
      std::vector<int> init;
      if (!greedy(init, rng, /*usePrev=*/r == 0 && !prevAssign.empty()) &&
          !greedy(init, rng, false))
        continue;
      Solution sol = localSearch(init, rng);
      if (sol.cost < best.cost - 1e-6)
        best = sol;
    }
    // Iterated local search: perturb the best solution (move a few random
    // free nodes to random feasible slots) and re-optimize; keep improvements.
    if (!best.slot.empty()) {
      std::vector<int> freeNodes;
      for (unsigned n = 0; n < N; ++n)
        if (pinned[n] < 0)
          freeNodes.push_back(n);
      for (int it = 0; it < perturbIters && !freeNodes.empty(); ++it) {
        std::vector<int> cand = best.slot;
        int k = 2 + it % 6;
        for (int j = 0; j < k; ++j) {
          int n = freeNodes[rng() % freeNodes.size()];
          cand[n] = rng() % slots.size();
        }
        std::vector<std::array<double, kNumRes>> load(slots.size());
        for (auto &l : load)
          l.fill(0);
        for (unsigned n = 0; n < N; ++n)
          for (int r = 0; r < kNumRes; ++r)
            load[cand[n]][r] += nodes[n].res[r];
        bool ok = true;
        for (unsigned s = 0; s < slots.size() && ok; ++s)
          for (int r = 0; r < kNumRes && ok; ++r)
            ok = load[s][r] <= rhoS[s] * slots[s].cap[r] + 1e-9;
        if (!ok)
          continue;
        Solution sol = localSearch(cand, rng);
        if (sol.cost < best.cost - 1e-6)
          best = sol;
      }
    }
    if (best.slot.empty()) {
      emitError(module.getLoc(), "floorplan infeasible: the design does not "
                                 "fit the slots at rho = ")
          << rho.getValue();
      return signalPassFailure();
    }

    // ---- annotate IR, emit Tcl and JSON
    auto *ctx = module.getContext();
    for (unsigned n = 0; n < N; ++n)
      for (Operation *op : nodes[n].ops)
        op->setAttr("hls.floorplan_slot",
                    StringAttr::get(ctx, slots[best.slot[n]].name));
    emitTcl(best.slot);

    llvm::json::Object asg, loads, out;
    std::vector<std::array<double, kNumRes>> load(slots.size());
    for (auto &l : load)
      l.fill(0);
    unsigned moved = 0;
    for (unsigned n = 0; n < N; ++n) {
      asg[nodes[n].key] = slots[best.slot[n]].name;
      for (int r = 0; r < kNumRes; ++r)
        load[best.slot[n]][r] += nodes[n].res[r];
      moved += prevSlot[n] >= 0 && prevSlot[n] != best.slot[n];
    }
    for (unsigned s = 0; s < slots.size(); ++s) {
      llvm::json::Object o;
      for (int r = 0; r < kNumRes; ++r)
        o[kResName[r]] = slots[s].cap[r] > 0 ? load[s][r] / slots[s].cap[r]
                                             : 0.0;
      loads[slots[s].name] = std::move(o);
    }
    double cut = 0;
    for (auto &e : edges)
      if (slots[best.slot[e.a]].slr != slots[best.slot[e.b]].slr)
        cut += e.w;
    llvm::json::Array rtlArr;
    for (auto &p : rtl)
      rtlArr.push_back(p);
    out["assignment"] = std::move(asg);
    out["slot_util"] = std::move(loads);
    llvm::json::Object rhoOut;
    for (unsigned s = 0; s < slots.size(); ++s)
      rhoOut[slots[s].name] = rhoS[s];
    out["slot_rho"] = std::move(rhoOut);
    out["slot_mode"] = mode;
    out["cost"] = best.cost;
    out["slr_cut_weight"] = cut;
    out["moved"] = (int64_t)moved;
    out["nodes"] = (int64_t)N;
    out["edges"] = (int64_t)edges.size();
    llvm::json::Array droppedArr;
    for (auto &d : dropped)
      droppedArr.push_back(d);
    out["dropped"] = std::move(droppedArr);
    out["action"] = action;
    out["rtl_passes"] = std::move(rtlArr);
    std::error_code ec;
    llvm::raw_fd_ostream js(outJson, ec);
    if (ec) {
      emitError(module.getLoc(), "cannot write " + outJson);
      return signalPassFailure();
    }
    js << llvm::formatv("{0:1}", llvm::json::Value(std::move(out))) << "\n";
  }
};

} // namespace

std::unique_ptr<Pass> scalehls::createPhysicalFloorplanPass() {
  return std::make_unique<PhysicalFloorplanPass>();
}
