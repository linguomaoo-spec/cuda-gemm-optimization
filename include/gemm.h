#pragma once

// CPU reference implementation
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K);

// Naive CUDA GEMM
void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K);

// Tiled shared-memory CUDA GEMM
void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K);
