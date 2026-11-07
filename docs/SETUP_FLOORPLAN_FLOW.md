# Setting up the floorplan-aware HIDA flow from scratch

Target setup: Windows PC with Vivado/Vitis HLS 2023.2, compiler in WSL2
Ubuntu on the same PC. Every command below was run while building and
testing the flow (Ubuntu 24.04, clang 18, cmake 3.28, ninja 1.11, 4 cores,
15 GB RAM), except the Vitis HLS and Vivado steps, which need the vendor
tools. Package status is called out where something is fragile.

## 0. Known package problems (read first)

| Component | Status | What to do |
|---|---|---|
| `./build-scalehls.sh` | Does not work on a fresh machine: the Polygeist submodule URL is SSH only, and it builds LLVM + Clang + MLIR in Debug (60+ GB disk, 32 GB RAM, many hours) | Use `tools/floorplan/setup_compiler.sh` (Release, MLIR only, HTTPS) |
| Polygeist / `cgeist` (C/C++ front-end) | Not built by the script; only needed for PolyBench C kernels | Build later with the repo's original instructions if you need C input |
| Torch-MLIR (`requirements.txt`) | Pins nightly wheels from Dec 2023 (`torch==2.2.0.dev20231204+cpu`, `torch-mlir==20231229.1067`); old nightlies disappear from the package index and they need Python 3.10 or 3.11, not 3.12+ | Only needed to turn a PyTorch model into `.mlir`. `samples/pytorch/resnet18/resnet18.mlir` is already in the repo, so start without it |
| OR-Tools | `pip install ortools` works (tested 9.15) | - |
| Vitis HLS / Vivado 2023.2 | Fine; the generated Tcl was written for this version | - |

## 1. Windows side

1. Install Vivado and Vitis HLS 2023.2 (you have them). Note the install path,
   for example `C:\Xilinx\Vivado\2023.2` and `C:\Xilinx\Vitis_HLS\2023.2`.
2. Install WSL2 with Ubuntu 24.04 from an administrator PowerShell:
   ```
   wsl --install -d Ubuntu-24.04
   ```
   Reboot, open Ubuntu, create your user.
3. Give WSL enough memory for the LLVM build. Create `C:\Users\<you>\.wslconfig`:
   ```
   [wsl2]
   memory=16GB
   processors=8
   swap=16GB
   ```
   then `wsl --shutdown` in PowerShell and reopen Ubuntu. 12 GB RAM is the
   practical minimum with one link job at a time; more cores shorten the build.
4. Decide where the work directory lives. Put it on the Windows drive so both
   sides see it, for example `C:\hida`, which WSL sees as `/mnt/c/hida`.
   Builds are faster on the Linux file system, so keep the compiler clone
   in `~/hida` and only the run directories under `/mnt/c/hida`.

## 2. Linux side: packages

```
sudo apt update
sudo apt install -y git cmake ninja-build clang lld python3 python3-pip python3-venv
clang --version   # 18.x on Ubuntu 24.04 is what was tested
```

## 3. Clone and build the compiler (about 1.5 h on 4 cores)

```
cd ~
git clone https://github.com/LouHaina/ScaleHLS-HIDA.git hida
cd hida
git checkout claude/fpga-floorplan-impl      # until PR #3 and PR #4 are merged into main
bash tools/floorplan/setup_compiler.sh -j $(nproc)
export PATH=$HOME/hida/build/bin:$PATH        # add to ~/.bashrc
```

What the script does, in case a step fails:

1. Clones `hanchenye/Polygeist` over HTTPS at the commit the repo pins
   (the submodule URL in `.gitmodules` is `git@github.com:...`, which needs an
   SSH key; the script avoids it).
2. Clones LLVM at the commit Polygeist pins (`cf0ff67a`, Nov 2022) with
   `--filter=blob:none` to keep the download around 1 GB.
3. Configures a Release build of LLVM + MLIR + ScaleHLS only (no Clang),
   host target only, `lld`, one link job at a time. Disk: about 4 GB for
   sources and 1 GB for the build directory.
4. Builds `scalehls-opt`, `scalehls-translate`, and the `FileCheck`,
   `count`, `not` utilities for the tests.

Check the build:

```
python3 polygeist/llvm-project/llvm/utils/lit/lit.py build/tools/scalehls/test
# expected: 43 passed, 2 unsupported (Python bindings and Polygeist are not built)
```

If `cmake` complains about the compiler, make sure `clang`, `clang++` and
`ld.lld` are on the PATH (`sudo apt install clang lld`). If a link step is
killed, the machine ran out of memory: raise `memory=` in `.wslconfig`.

## 4. Python tools

```
cd ~/hida
python3 -m pip install --user ortools      # or inside a venv
python3 tools/floorplan/floorplan.py --help
```

Quick self-test on the shipped ResNet18 C++ (no compiler needed):

```
python3 tools/floorplan/proto_floorplan.py samples/pytorch/resnet18/resnet18.cpp \
    --top forward --slrs 3 --baseline-tcl data/run_0_2_1/floorplan_constraints.tcl
python3 tools/floorplan/parse_reports.py data/run_0_2_1 | head -40
```

## 5. One manual pass of the flow (compiler side)

```
mkdir -p /mnt/c/hida/runs/r0 && cd ~/hida
OPTS="top-func=forward loop-tile-size=4 loop-unroll-factor=2"     # scale these up later

# 5a. compile once to export the graph
scalehls-opt samples/pytorch/resnet18/resnet18.mlir \
    -hida-pytorch-pipeline="$OPTS floorplan-graph-output=/mnt/c/hida/runs/r0/graph.json" > /dev/null

# 5b. solve the SLR assignment (3 SLRs on U280)
python3 tools/floorplan/floorplan.py solve /mnt/c/hida/runs/r0/graph.json \
    --device tools/floorplan/devices/u280.json --balance 0.3 --out /mnt/c/hida/runs/r0/assignment.json

# 5c. compile again with the assignment and emit the C++
scalehls-opt samples/pytorch/resnet18/resnet18.mlir \
    -hida-pytorch-pipeline="$OPTS floorplan-graph-output=/mnt/c/hida/runs/r0/graph.json floorplan-assignment=/mnt/c/hida/runs/r0/assignment.json" \
  | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives > /mnt/c/hida/runs/r0/design.cpp

# 5d. Vivado constraints and the cell manifest
python3 tools/floorplan/floorplan.py emit /mnt/c/hida/runs/r0/assignment.json \
    --device tools/floorplan/devices/u280.json \
    --out-tcl /mnt/c/hida/runs/r0/floorplan.tcl --out-manifest /mnt/c/hida/runs/r0/manifest.json
grep -c "inline off" /mnt/c/hida/runs/r0/design.cpp    # 24 for ResNet18: one per unit
```

Note that `config.json` in the repository root is read by QoR estimation
(default `target-spec=./config.json`), so run `scalehls-opt` from the repo
root or pass `target-spec=/path/to/config.json`.

## 6. Vitis HLS and Vivado (Windows side)

From a Windows command prompt, or from WSL through the `.bat` launchers:

```
C:\Xilinx\Vitis_HLS\2023.2\bin\vitis_hls.bat -f C:\hida\tools\floorplan\hls_csynth.tcl ^
    -tclargs C:\hida\runs\r0\design.cpp C:\hida\runs\r0\hls forward xcu280-fsvh2892-2L-e 4.0
```

This produces `runs\r0\hls\forward_csynth.rpt` (per-function LUT/FF/DSP/BRAM)
and the exported RTL under `runs\r0\hls\hls_prj\sol1\impl`. Then calibrate
and re-solve on the Linux side:

```
python3 tools/floorplan/floorplan.py calibrate /mnt/c/hida/runs/r0/graph.json \
    --hls-dir /mnt/c/hida/runs/r0/hls --out /mnt/c/hida/runs/r0/calibration.json
python3 tools/floorplan/floorplan.py solve /mnt/c/hida/runs/r0/graph.json --device tools/floorplan/devices/u280.json \
    --calibration /mnt/c/hida/runs/r0/calibration.json --out /mnt/c/hida/runs/r0/assignment.json
# then repeat 5c and 5d
```

Synthesize the RTL to a checkpoint once (your existing `post_floorplan_ooc.dcp`
was made this way; `synth_design -mode out_of_context -flatten_hierarchy rebuilt`),
then run one P&R iteration:

```
C:\Xilinx\Vivado\2023.2\bin\vivado.bat -mode batch -source C:\hida\tools\floorplan\run_vivado_iter.tcl ^
    -tclargs C:\hida\runs\r0\post_synth.dcp C:\hida\runs\r0\floorplan.tcl 4.0 C:\hida\runs\r0\out
```

The Tcl writes `timing_summary.rpt`, `utilization_slr.rpt`, `congestion.rpt`,
`slr_crossing.rpt`, `crossing_paths.csv` and `post_route.dcp` into `out\`,
and prints `unit <name>: N cells -> SLRk` for each unit. If it prints
`WARNING: floorplan units with no matching cells`, the RTL hierarchy does not
match the manifest regexes; open the checkpoint and check the instance names
with `get_cells -hierarchical -filter {NAME =~ *forward_node*}`.

Parse the feedback:

```
python3 tools/floorplan/parse_reports.py /mnt/c/hida/runs/r0/out --out /mnt/c/hida/runs/r0/feedback.json
python3 tools/floorplan/floorplan.py solve ... --feedback /mnt/c/hida/runs/r0/feedback.json \
    --prev /mnt/c/hida/runs/r0/assignment.json --out /mnt/c/hida/runs/r1/assignment.json
```

## 7. Automating the loop

Copy `tools/floorplan/loop_config.example.json`, set `workdir` to
`/mnt/c/hida/runs/resnet18`, and point `hls` and `vivado` at the `.bat`
launchers through WSL, for example:

```
"hls":    "/mnt/c/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls.bat -f tools/floorplan/hls_csynth.tcl -tclargs {cpp} {hls_dir}",
"vivado": "/mnt/c/Xilinx/Vivado/2023.2/bin/vivado.bat -mode batch -source tools/floorplan/run_vivado_iter.tcl -tclargs {dcp} {tcl} {period} {out} {prev}"
```

Paths passed to the Windows tools must be Windows paths; `wslpath -w` converts
them. The `dcp` entry must point at a post-synthesis checkpoint, so add a
`synth.tcl` step (synth_design on the exported RTL) to your `hls` command, or
run it once by hand and reference the file. Then:

```
python3 tools/floorplan/floorplan.py loop my_loop_config.json
```

The driver stops at WNS >= 0, after `iters` iterations, or when nothing
changes, and prints the best iteration. Every iteration keeps its graph,
assignment, Tcl, manifest and parsed feedback under `workdir/iter_N/`.

## 8. Getting other models into the flow

Only `resnet18.mlir` ships with the repository. For VGG16, MobileNet, YOLO
you need Torch-MLIR. Use a separate Python 3.10 or 3.11 environment:

```
sudo apt install -y python3.11 python3.11-venv
python3.11 -m venv ~/mlir_venv && source ~/mlir_venv/bin/activate
pip install --upgrade pip
pip install --no-deps -r requirements.txt     # may fail if the pinned nightlies are gone
```

If the pinned wheels are no longer on the index, install the newest
`torch-mlir` wheel that still exports `torch_mlir.compile` with
`OutputType.LINALG_ON_TENSORS` (the sample scripts use that API) and adjust
`samples/pytorch/<model>/<model>.py` accordingly; the generated `.mlir` must
be in the linalg dialect with static shapes. Then:

```
cd samples/pytorch/vgg16 && python3 vgg16.py > vgg16.mlir
```

## 9. Scaling the benchmark

The design must need more than one SLR before floorplanning can pay off.
Sweep `loop-unroll-factor` 4, 8, 16 and `loop-tile-size` 4, 8 and read the
total LUT/DSP from `forward_csynth.rpt` until the design uses 40 to 70 percent
of the device (U280: about 1.3 M LUT, 9 k DSP). Record vanilla HIDA at that
size first (no `floorplan.tcl`, same period and placer directives); that is
the baseline the thesis has to beat.
