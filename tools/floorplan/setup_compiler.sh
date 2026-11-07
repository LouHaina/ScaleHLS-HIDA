#!/usr/bin/env bash
# Build scalehls-opt / scalehls-translate (and FileCheck for the tests) from a
# fresh clone, without SSH keys and without the Clang/Polygeist front-end.
# This is the exact sequence that was used to build and test the floorplan
# passes (Ubuntu 24.04, clang 18, cmake 3.28, ninja 1.11, 4 cores, 15 GB RAM,
# about 1.5 h). Usage, from the repository root:
#   bash tools/floorplan/setup_compiler.sh [-j JOBS]
set -euo pipefail
JOBS=$(nproc)
while getopts 'j:' opt; do case $opt in j) JOBS=$OPTARG;; esac; done
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# 1. Polygeist submodule: the .gitmodules URL is SSH (git@github.com:...),
#    which fails without a GitHub SSH key, so clone it over HTTPS by hand at
#    the pinned commit. --filter=blob:none keeps the download small.
PG_SHA=$(git ls-tree HEAD polygeist | awk '{print $3}')
if [ ! -f polygeist/CMakeLists.txt ]; then
  rm -rf polygeist
  git clone --filter=blob:none https://github.com/hanchenye/Polygeist.git polygeist
fi
git -C polygeist checkout -q "$PG_SHA"

# 2. LLVM at the commit Polygeist pins (hanchenye/llvm-project fork; the same
#    commit also exists in llvm/llvm-project).
LLVM_SHA=$(git -C polygeist ls-tree HEAD llvm-project | awk '{print $3}')
if [ ! -f polygeist/llvm-project/llvm/CMakeLists.txt ]; then
  rm -rf polygeist/llvm-project
  git clone --filter=blob:none --no-checkout https://github.com/hanchenye/llvm-project.git polygeist/llvm-project \
    || git clone --filter=blob:none --no-checkout https://github.com/llvm/llvm-project.git polygeist/llvm-project
fi
git -C polygeist/llvm-project checkout -q "$LLVM_SHA"

# 3. Configure: Release, MLIR only (no clang), host target only, lld linker,
#    one link job at a time (links are the memory peak).
mkdir -p build && cd build
if [ ! -f CMakeCache.txt ]; then
  cmake -G Ninja ../polygeist/llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="scalehls" \
    -DLLVM_EXTERNAL_SCALEHLS_SOURCE_DIR="$ROOT" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_TERMINFO=OFF
fi

# 4. Build only what the flow needs (plus the lit utilities for the tests).
ninja -j"$JOBS" scalehls-opt scalehls-translate FileCheck count not

echo
echo "Built: $ROOT/build/bin/scalehls-opt and scalehls-translate"
echo "Add to PATH:  export PATH=$ROOT/build/bin:\$PATH"
echo "Run tests:    python3 polygeist/llvm-project/llvm/utils/lit/lit.py build/tools/scalehls/test"
