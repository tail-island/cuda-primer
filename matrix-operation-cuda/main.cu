#include <iostream>
#include <vector>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../util/cudautil.hpp"
#include "../util/util.hpp"

const std::size_t M = 1001;
const std::size_t N = 1002;
const std::size_t K = 1003;

__global__
void matmul(const float* matrix_a, const float* matrix_b, float* matrix_c, std::size_t size_m, std::size_t size_n, std::size_t size_k) {
    const auto i = blockIdx.y * blockDim.y + threadIdx.y;
    const auto j = blockIdx.x * blockDim.x+ threadIdx.x;

    if (i >= size_m || j >= size_n) {
        return;
    }

    for (auto k = static_cast<std::size_t>(0); k < size_k; ++k) {
        matrix_c[i * size_n + j] += matrix_a[i * size_k + k] * matrix_b[k * size_n + j];
    }
}

int main(int argc, char** argv) {
    const auto a = thrust::host_vector<float>(util::linspace(0.0f, 1.0f, M * K));
    const auto a_device = thrust::device_vector<float>(a);

    const auto b = thrust::host_vector<float>(util::linspace(0.0f, 1.0f, K * N));
    const auto b_device = thrust::device_vector<float>(b);

    cuda_check(cudaDeviceSynchronize());

    util::timeit([&]() {
        auto c_device = thrust::device_vector<float>(M * N, 0.0f);

        matmul<<<dim3((M + 16 - 1) / 16, (N + 16 - 1) / 16), dim3(16, 16)>>>(a_device.data().get(), b_device.data().get(), c_device.data().get(), M, N, K);
        cuda_check(cudaGetLastError());

        cuda_check(cudaDeviceSynchronize());
    });

    auto c_device = thrust::device_vector<float>(M * N, 0.0f);

    matmul<<<dim3((M + 16 - 1) / 16, (N + 16 - 1) / 16), dim3(16, 16)>>>(a_device.data().get(), b_device.data().get(), c_device.data().get(), M, N, K);
    cuda_check(cudaGetLastError());

    const auto c = thrust::host_vector<float>(c_device);

    std::cout << c[0        ] << std::endl;
    std::cout << c[1        ] << std::endl;
    std::cout << c[2        ] << std::endl;

    std::cout << c[M * N - 3] << std::endl;
    std::cout << c[M * N - 2] << std::endl;
    std::cout << c[M * N - 1] << std::endl;

    return 0;
}

// 以下は、遅い上にバグがある駄目コード。

// __global__
// void matmul(float* matrix_a, float* matrix_b, float* matrix_c, std::size_t size_m, std::size_t size_n, std::size_t size_k) {
//     const auto i = blockIdx.y * blockDim.y + threadIdx.y;
//     const auto k = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i >= size_m || k >= size_k) {
//         return;
//     }

//     for (auto j = static_cast<std::size_t>(0); j < size_n; ++j) {
//         matrix_c[i * size_n + j] += matrix_a[i * size_k + k] * matrix_b[k * size_n + j];
//     }
// }

// int main(int argc, char** argv) {
//     auto a = thrust::host_vector<float>(util::linspace(0.0f, 1.0f, M * K));
//     auto a_device = thrust::device_vector<float>(a);

//     auto b = thrust::host_vector<float>(util::linspace(0.0f, 1.0f, K * N));
//     auto b_device = thrust::device_vector<float>(b);

//     cuda_check(cudaDeviceSynchronize());

//     util::timeit([&]() {
//         auto c_device = thrust::device_vector<float>(M * N, 0.0f);

//         matmul<<<dim3((M + 16 - 1) / 16, (K + 16 - 1) / 16), dim3(16, 16)>>>(a_device.data().get(), b_device.data().get(), c_device.data().get(), M, N, K);
//         cuda_check(cudaGetLastError());

//         cuda_check(cudaDeviceSynchronize());
//     });

//     auto c_device = thrust::device_vector<float>(M * N, 0.0f);

//     matmul<<<dim3((M + 16 - 1) / 16, (K + 16 - 1) / 16), dim3(16, 16)>>>(a_device.data().get(), b_device.data().get(), c_device.data().get(), M, N, K);
//     cuda_check(cudaGetLastError());

//     auto c = thrust::host_vector<float>(c_device);

//     std::cout << c[0        ] << std::endl;
//     std::cout << c[1        ] << std::endl;
//     std::cout << c[2        ] << std::endl;

//     std::cout << c[M * N - 3] << std::endl;
//     std::cout << c[M * N - 2] << std::endl;
//     std::cout << c[M * N - 1] << std::endl;

//     return 0;
// }
