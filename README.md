# cuda-gemm-optimization

This repository contains a mini CUDA GEMM optimization project designed as an interview ‑style portfolio piece rather than a research paper. The goal is to demonstrate understanding of the CUDA programming model, GPU memory hierarchy, kernel profiling and performance optimization techniques, and to present measurable improvements between a nïve and optimized implementation.

## Goals

- Understand the CUDA execution model (thread/block/grid hierarchy and occupancy).
- Optimize memory access patterns using shared memory tiling and memory coalescing.
- Benchmark multiple GEMM implementations and interpret profiling results.
- Document optimization strategies and explain the performance gains.

## Implementations

The project provides several matrix multiplication implementations:

| Implementation | Description |
| --- | --- |
| **CPU Reference (`gemm_cpu`)** | Plain C++ triple‑nested loop used to verify correctness. |
| **Naïve CUDA (`gemm_naive`)** | Each thread computes one output element without shared memory or tiling. |
| **Tiled CUDA (`gemm_tiled`)** | Uses `TILE_SIZE×TILE_SIZE` blocks and shared memory to improve data reuse and reduce global memory traffic. |

Future versions may explore additional optimizations such as loop unrolling, different block sizes, vectorized loads and WMMA/Tensor Core intrinsics.

## Benchmark Results

The `scripts/run_bench.sh` script (to be added) will compile and benchmark the implementations over a few matrix sizes. An example of the expected output (times are illustrative):

| Version | Size (M×N×K) | Time (ms) | GFLOPS | Speedup |
| --- | --- | ---: | ---: | ---: |
| Naïve | 1024×1024×1024 | 250.0 | 8.4 | 1.0× |
| Tiled | 1024×1024×1024 | 75.0 | 28.0 | 3.3× |

GFLOPS is computed as `2 * M * N * K / (time_ms / 1000) / 1e9`. Your actual numbers will vary depending on GPU hardware.

## Optimization Notes

- **Global memory bottlenecks:** The nïve kernel repeatedly reads the same matrix elements from global memory, leading to poor arithmetic intensity.
- **Shared memory tiling:** By loading sub‑matrices of `A` and `B` into fast on‑chip shared memory, threads within a block can reuse data and dramatically reduce the number of global memory accesses.
- **Memory coalescing:** Properly aligned loads (each warp accessing contiguous memory) ensure efficient use of the memory bus.
- **Block size tuning:** `TILE_SIZE` determines how many threads participate in a block and thus influences occupancy and shared memory usage. Try experimenting with 16, 32 and other values.

More detailed profiling results (using `nvprof`/`nsys`/`ncu`) can be recorded in `docs/optimization_notes.md`.

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│  ├── gemm.h          # Function declarations
├── src/
│  ├── main.cu         # Benchmark harness
│  ├── gemm_naive.cu   # Naïve CUDA kernel
│  ├── gemm_tiled.cu   # Shared‑memory tiled kernel
│  └── gemm_utils.cu   # CPU reference implementation
├── tests/
│  └── test_correctness.cu   # (to be implemented) correctness tests
├── scripts/
│  └── run_bench.sh    # (to be implemented) benchmarking script
├── docs/
│  └── optimization_notes.md # (to be written) deeper analysis
└── results/
   └── benchmark.md    # (placeholder) store benchmark results
```

## Future Work

- Implement additional optimizations (warp‑level primitives, vectorized loads, half‑precision computation).
- Add an autotuner to explore block sizes and tile dimensions automatically.
- Compare performance against `cuBLAS` and document the gap.
- Extend the test suite under `tests/` to validate correctness across random sizes.
