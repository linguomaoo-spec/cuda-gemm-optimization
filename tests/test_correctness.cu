#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "../include/gemm.h"

bool compare_matrices(const float *ref, const float *test, int size, float tol = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        if (fabs(ref[i] - test[i]) > tol) {
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 128;
    const int N = 128;
    const int K = 128;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_naive(M * N);
    std::vector<float> h_C_tiled(M * N);

    // initialize inputs with random floats
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // copy inputs to device
    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    // run naive kernel
    gemm_naive(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C_naive.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    // run tiled kernel
    gemm_tiled(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C_tiled.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    // compute CPU reference
    gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    bool naive_ok = compare_matrices(h_C_cpu.data(), h_C_naive.data(), M * N);
    bool tiled_ok = compare_matrices(h_C_cpu.data(), h_C_tiled.data(), M * N);

    std::cout << "Naive kernel correctness: " << (naive_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Tiled kernel correctness: " << (tiled_ok ? "PASSED" : "FAILED") << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (naive_ok && tiled_ok) ? 0 : 1;
}
