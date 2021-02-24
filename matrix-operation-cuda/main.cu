#include <iostream>
#include <vector>

#include <cuda.h>

#include "../util/cudautil.hpp"
#include "../util/util.hpp"

const std::size_t M = 1001;
const std::size_t N = 1002;
const std::size_t K = 1003;

__global__
void matmul(float* matrix_a, float* matrix_b, float* matrix_c, std::size_t size_m, std::size_t size_n, std::size_t size_k) {
    const auto i = blockDim.y * blockIdx.y + threadIdx.y;
    const auto j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= size_m || j >= size_n) {
        return;
    }

    matrix_c[i * size_n + j] = 0.0f;

    for (auto k = static_cast<std::size_t>(0); k < size_k; ++k) {
        matrix_c[i * size_n + j] += matrix_a[i * size_k + k] * matrix_b[k * size_n + j];
    }
}

int main(int argc, char** argv) {
    auto a = util::linspace(0.0f, 1.0f, M * K);

    float* a_device;
    cuda_check(cudaMalloc(&a_device, sizeof(float) * M * K));
    cuda_check(cudaMemcpy(a_device, a.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));

    auto b = util::linspace(0.0f, 1.0f, K * N);

    float* b_device;
    cuda_check(cudaMalloc(&b_device, sizeof(float) * K * N));
    cuda_check(cudaMemcpy(b_device, b.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice));

    float* c_device;
    cuda_check(cudaMalloc(&c_device, sizeof(float) * M * N));

    cuda_check(cudaDeviceSynchronize());

    util::timeit([&]() {
        matmul<<<dim3((M + 16 - 1) / 16, (N + 16 - 1) / 16), dim3(16, 16)>>>(a_device, b_device, c_device, M, N, K);
        cuda_check(cudaGetLastError());

        cuda_check(cudaDeviceSynchronize());
    });

    matmul<<<dim3((M + 16 - 1) / 16, (N + 16 - 1) / 16), dim3(16, 16)>>>(a_device, b_device, c_device, M, N, K);
    cuda_check(cudaGetLastError());

    auto c = std::vector<float>(M * N, 0.0f);
    cuda_check(cudaMemcpy(c.data(), c_device, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    cuda_check(cudaFree(c_device));
    cuda_check(cudaFree(b_device));
    cuda_check(cudaFree(a_device));

    std::cout << c[0        ] << std::endl;
    std::cout << c[1        ] << std::endl;
    std::cout << c[2        ] << std::endl;

    std::cout << c[M * N - 3] << std::endl;
    std::cout << c[M * N - 2] << std::endl;
    std::cout << c[M * N - 1] << std::endl;

    return 0;
}
