# Optimization Notes

This document summarizes the key optimization techniques explored in this project, along with observations from profiling.

## Global Memory vs. Shared Memory

The naive GEMM implementation loads each matrix element from global memory every time it is used, leading to high latency and low arithmetic intensity. By tiling the matrices and staging blocks of **A** and **B** into **shared memory**, we reuse data across multiple threads within a block, reducing global memory traffic and improving arithmetic intensity.

## Memory Coalescing

For efficient global memory loads, threads in a warp should access consecutive memory addresses. In the tiled kernel, each thread reads contiguous elements from the global arrays into the shared‑memory tiles. Properly aligning the row‑major matrices and choosing appropriate tile sizes helps achieve coalesced accesses and maximize bandwidth utilization.

## Bank Conflicts

Shared memory is divided into banks. If multiple threads access addresses in the same bank, bank conflicts occur and serialization reduces performance. We analyzed the access pattern in the tiled kernel and found that with a tile size of 16 the default indexing avoids most conflicts. Adjusting the tile dimensions or adding padding can further mitigate conflicts if necessary.

## Loop Unrolling

Unrolling the innermost accumulation loop (e.g. `for (int k = 0; k < TILE_SIZE; ++k)`) can reduce loop overhead and enable better instruction‑level parallelism. In experiments with `#pragma unroll`, we observed modest improvements for certain problem sizes.

## Block and Thread Configuration

Experimenting with different block sizes (such as 16x16, 32x8, and 32x16) revealed trade‑offs between occupancy and per‑thread workload. For our GPU, a 16×16 block with a tile size of 16 provided a good balance between resource usage and performance.

## Profiling Observations

Profiling tools such as `nvprof`, `nsys`, or `ncu` were used to analyze kernel performance. Key metrics include:

- **Global memory throughput** increased significantly after introducing shared memory tiling.
- **Compute utilization** improved when we tuned block sizes and applied loop unrolling.
- **Kernel occupancy** remained above 60%, indicating good latency hiding.
- The primary bottleneck in the naive kernel was global memory latency, whereas in the tiled kernel the bottleneck shifted to compute.

See `results/benchmark.md` for detailed timing and GFLOPS data.
