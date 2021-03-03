#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <thrust/device_vector.h>

#include "../util/cudautil.hpp"
#include "../util/util.hpp"

const std::size_t M = 1001;
const std::size_t N = 1002;
const std::size_t K = 1003;

template <typename T>
inline auto transpose(const std::vector<T>& matrix, std::size_t row_size, std::size_t col_size) noexcept {
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

    const auto a = transpose(util::linspace(0.0f, 1.0f, M * K), M, K);
    auto a_device = thrust::device_vector<float>(M * K);
    cublas_check(cublasSetMatrix(M, K, sizeof(float), a.data(), M, a_device.data().get(), M));

    const auto b = transpose(util::linspace(0.0f, 1.0f, K * N), K, N);
    auto b_device = thrust::device_vector<float>(K * N);
    cublas_check(cublasSetMatrix(K, N, sizeof(float), b.data(), K, b_device.data().get(), K));

    cuda_check(cudaDeviceSynchronize());

    util::timeit([&]() {
        auto c_device = thrust::device_vector<float>(M * N);

        const auto alpha = 1.0f;
        const auto beta  = 0.0f;

        cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a_device.data().get(), M, b_device.data().get(), K, &beta, c_device.data().get(), M));

        cuda_check(cudaDeviceSynchronize());
    });

    auto c_device = thrust::device_vector<float>(M * N);

    const auto alpha = 1.0f;
    const auto beta  = 0.0f;

    cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a_device.data().get(), M, b_device.data().get(), K, &beta, c_device.data().get(), M));

    auto c = std::vector<float>(N * M, 0.0f);
    cublas_check(cublasGetMatrix(M, N, sizeof(float), c_device.data().get(), M, c.data(), M));

    cublas_check(cublasDestroy(handle));

    std::cout << c[0 * M] << std::endl;
    std::cout << c[1 * M] << std::endl;
    std::cout << c[2 * M] << std::endl;

    std::cout << c[N * M - 1 - 2 * M] << std::endl;
    std::cout << c[N * M - 1 - 1 * M] << std::endl;
    std::cout << c[N * M - 1 - 0 * M] << std::endl;

    return 0;
}
