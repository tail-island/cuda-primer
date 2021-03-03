#include <iostream>
#include <iterator>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <thrust/complex.h>

#include "../util/cudautil.hpp"
#include "../util/util.hpp"

__device__
inline auto divergence_count(const thrust::complex<float>& c) noexcept {
    auto z = thrust::complex(0.0f, 0.0f);

    for (auto i = 0; i < 100; ++i) {
        if (std::isinf(z.real())) {
            return i;
        }

        z = z * z + c;
    }

    return 0;
}

__global__
void mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size, int* result) {
    const auto i = blockDim.y * blockIdx.y + threadIdx.y;
    const auto j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= imag_size || j >= real_size) {  // imag_indexがimag_sizeを超えることはないのですけど、パターン化しておいたほうがバグが減るので。
        return;
    }

    const auto imag = imag_min + (imag_max - imag_min) / static_cast<float>(imag_size - 1) * static_cast<float>(i);
    const auto real = real_min + (real_max - real_min) / static_cast<float>(real_size - 1) * static_cast<float>(j);

    result[i * real_size + j] = divergence_count(thrust::complex(real, imag));
}

inline auto mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size) noexcept {
    auto result = std::vector<int>(real_size * imag_size);

    int* result_device;
    cuda_check(cudaMalloc(&result_device, sizeof(int) * real_size * imag_size));

    const auto grid  = dim3((real_size + 256 - 1) / 256, imag_size);
    const auto block = dim3(256, 1);
    mandelbrot_set<<<grid, block>>>(real_min, real_max, real_size, imag_min, imag_max, imag_size, result_device);
    cuda_check(cudaGetLastError());

    cuda_check(cudaMemcpy(result.data(), result_device, sizeof(int) * real_size * imag_size, cudaMemcpyDeviceToHost));

    cuda_check(cudaFree(result_device));

    return std::make_tuple(result, real_size, imag_size);
}

inline auto mandelbrot_set(float real_min, float real_max, float imag_min, float imag_max) noexcept {
    const auto [real_size, imag_size] = [&]() {
        const auto real_diff = real_max - real_min;
        const auto imag_diff = imag_max - imag_min;

        return std::make_tuple(
            static_cast<std::size_t>(1000 * std::min(real_diff / imag_diff, 1.0f)),
            static_cast<std::size_t>(1000 * std::min(imag_diff / real_diff, 1.0f))
        );
    }();

    return mandelbrot_set(real_min, real_max, real_size, imag_min, imag_max, imag_size);
}

int main(int argc, char** argv) {
    first_cudaMalloc_is_too_slow();
    cuda_check(cudaDeviceSynchronize());

    util::timeit([&]() {
        mandelbrot_set(-2.0f, 2.0f, -2.0f, 2.0f);
    });

    const auto [v, w, h] = mandelbrot_set(-2.0f, 2.0f, -2.0f, 2.0f);

    {
        auto it = std::begin(v);

        for (auto i = 0_z; i < h; ++i, it += w) {
            std::copy(it, it + w, std::ostream_iterator<float>(std::cout, "\t"));
            std::cout << std::endl;
        }
    }

    return 0;
}
