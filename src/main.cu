#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <cmath>
#include "gemm.h"

int main() {
    int M = 512;
    int N = 512;
    int K = 512;

    size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
    size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
    size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_ref = (float*)malloc(sizeC);

    // Initialize host matrices with random values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Launch naive kernel
    gemm_naive(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Compute reference result on CPU and compute max error
    gemm_cpu(h_A, h_B, h_ref, M, N, K);
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_error = std::max(max_error, std::abs(h_C[i] - h_ref[i]));
    }
    std::cout << "Max error: " << max_error << std::endl;

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);
    return 0;
}
