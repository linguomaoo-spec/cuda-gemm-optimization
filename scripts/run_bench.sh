#!/bin/bash
# Benchmark script for CUDA GEMM project
set -e

# Create build directory
BUILD_DIR="build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Configure and build
cd "$BUILD_DIR"
cmake ..
make -j

# Run the executable for various problem sizes and append results
sizes=(256 512 1024 2048)

for n in "${sizes[@]}"; do
    echo "Running GEMM for size ${n}x${n}x${n}..." | tee -a ../results/benchmark.md
    ./gemm $n $n $n | tee -a ../results/benchmark.md
    echo "" | tee -a ../results/benchmark.md
done

echo "Benchmark complete. Results stored in results/benchmark.md"
