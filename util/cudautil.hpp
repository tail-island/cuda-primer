#include <cstdlib>

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

inline auto first_cudaMalloc_is_too_slow() noexcept {
    int* i;
    cudaMalloc(&i, sizeof(int) * 1);
    cudaFree(i);
}

inline auto cuda_check(const cudaError_t& error) noexcept {
    if (error == cudaSuccess) {
        return;
    }

    std::cerr << cudaGetErrorString(error) << std::endl;
    std::exit(1);
}

inline auto curand_check(const curandStatus_t& status) noexcept {
    if (status == CURAND_STATUS_SUCCESS) {
        return;
    }

    std::cerr << "cuRAND error." << std::endl;
    std::exit(1);
}

inline auto cublas_check(const cublasStatus_t& status) noexcept {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }

    std::cerr << "cuBLAS error." << std::endl;
    std::exit(1);
}
