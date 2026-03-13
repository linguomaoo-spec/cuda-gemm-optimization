# Benchmark Results

The following table summarizes the execution time and computed GFLOPS for different matrix sizes. Replace the placeholder values with actual results from running `scripts/run_bench.sh`.

| Size (M=N=K) | Kernel Version | Time (ms) | GFLOPS | Speedup |
| --- | --- | ---:| ---:| ---:|
| 256 | Naive | TBD | TBD | 1.0x |
| 256 | Tiled | TBD | TBD | TBD |
| 512 | Naive | TBD | TBD | 1.0x |
| 512 | Tiled | TBD | TBD | TBD |
| 1024 | Naive | TBD | TBD | 1.0x |
| 1024 | Tiled | TBD | TBD | TBD |
| 2048 | Naive | TBD | TBD | 1.0x |
| 2048 | Tiled | TBD | TBD | TBD |

You can compute GFLOPS using the formula:

```
GFLOPS = 2 * M * N * K / (time_ms / 1000) / 1e9
```

where `time_ms` is the kernel execution time in milliseconds.
