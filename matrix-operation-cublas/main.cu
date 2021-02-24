#include <iostream>
#include <vector>

#include <cuda.h>
#include <cublas_v2.h>

#include "../util/cudautil.hpp"
#include "../util/util.hpp"

const std::size_t M = 1001;
const std::size_t N = 1002;
const std::size_t K = 1003;

template <typename T>
inline auto transpose(std::vector<T> matrix, std::size_t row_size, std::size_t col_size) noexcept {
    auto result = std::vector<T>(col_size * row_size);

    for (auto i = 0_z; i < col_size; ++i) {
        for (auto j = 0_z; j < row_size; ++j) {
            result[i * row_size + j] = matrix[j * col_size + i];
        }
    }

    return result;
}

int main(int argc, char** argv) {
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle));

    auto a = transpose(util::linspace(0.0f, 1.0f, M * K), M, K);

    float* a_device;
    cuda_check(cudaMalloc(&a_device, sizeof(float) * M * K));
    cublas_check(cublasSetMatrix(M, K, sizeof(float), a.data(), M, a_device, M));

    auto b = transpose(util::linspace(0.0f, 1.0f, K * N), K, N);

    float* b_device;
    cuda_check(cudaMalloc(&b_device, sizeof(float) * K * N));
    cublas_check(cublasSetMatrix(K, N, sizeof(float), b.data(), K, b_device, K));

    float* c_device;
    cuda_check(cudaMalloc(&c_device, sizeof(float) * M * N));

    cuda_check(cudaDeviceSynchronize());

    auto alpha = 1.0f;
    auto beta  = 0.0f;

    util::timeit([&]() {
        cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a_device, M, b_device, K, &beta, c_device, M));

        cuda_check(cudaDeviceSynchronize());
    });

    cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a_device, M, b_device, K, &beta, c_device, M));

    auto c = std::vector<float>(N * M, 0.0f);
    cublas_check(cublasGetMatrix(M, N, sizeof(float), c_device, M, c.data(), M));

    cuda_check(cudaFree(c_device));
    cuda_check(cudaFree(b_device));
    cuda_check(cudaFree(a_device));

    cublas_check(cublasDestroy(handle));

    std::cout << c[0 * M] << std::endl;
    std::cout << c[1 * M] << std::endl;
    std::cout << c[2 * M] << std::endl;

    std::cout << c[N * M - 1 - 2 * M] << std::endl;
    std::cout << c[N * M - 1 - 1 * M] << std::endl;
    std::cout << c[N * M - 1 - 0 * M] << std::endl;

    return 0;
}
