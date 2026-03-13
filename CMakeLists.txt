cmake_minimum_required(VERSION 3.8)
project(cuda_gemm_optimization LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(SRC_FILES
    src/main.cu
    src/gemm_naive.cu
    src/gemm_tiled.cu
    src/gemm_utils.cu
)

include_directories(include)

add_executable(gemm ${SRC_FILES})
