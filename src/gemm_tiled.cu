#include <cuda_runtime.h>
#include "gemm.h"

#define TILE_SIZE 16

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;
        // Load data into shared memory or zero if out of bounds
        if (row < M && tiledCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tiledCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (col < N && tiledRow < K) {
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        // Compute partial product for this tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
