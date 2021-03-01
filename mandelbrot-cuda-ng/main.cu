#include <complex>
#include <iostream>
#include <iterator>
#include <tuple>
#include <vector>

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

__device__
inline auto linspace(float start, float stop, std::size_t size) noexcept {
    auto result = new float[size];

    auto delta = (stop - start) / static_cast<float>(size - 1);

    for (auto i = static_cast<std::size_t>(0); i < size; ++i) {
        result[i] = start + delta * i;
    }

    return result;
}

__global__
void mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size, int* result) {
    auto imags = linspace(imag_min, imag_max, imag_size);
    auto reals = linspace(real_min, real_max, real_size);

    for (auto i = 0; i < imag_size; ++i) {
        for (auto j = 0; j < real_size; ++j) {
            result[i * real_size + j] = divergence_count(thrust::complex(reals[j], imags[i]));
        }
    }

    delete reals;
    delete imags;
}

inline auto mandelbrot_set(float real_min, float real_max, float imag_min, float imag_max) noexcept {
    auto [real_size, imag_size] = [&]() {
        auto real_diff = real_max - real_min;
        auto imag_diff = imag_max - imag_min;

        return std::make_tuple(
            static_cast<std::size_t>(1024 * std::min(real_diff / imag_diff, 1.0f)),
            static_cast<std::size_t>(1024 * std::min(imag_diff / real_diff, 1.0f))
        );
    }();

    auto result = std::vector<int>(real_size * imag_size);

    int* result_device;
    cuda_check(cudaMalloc(&result_device, sizeof(int) * real_size * imag_size));

    mandelbrot_set<<<1, 1>>>(real_min, real_max, real_size, imag_min, imag_max, imag_size, result_device);
    cuda_check(cudaGetLastError());

    cuda_check(cudaMemcpy(result.data(), result_device, sizeof(int) * real_size * imag_size, cudaMemcpyDeviceToHost));

    cuda_check(cudaFree(result_device));

    return std::make_tuple(result, real_size, imag_size);
}

int main(int argc, char** argv) {
    first_cudaMalloc_is_too_slow();
    cuda_check(cudaDeviceSynchronize());

    util::timeit([&]() {
        mandelbrot_set(-2.0f, 2.0f, -2.0f, 2.0f);
    });

    auto [v, w, h] = mandelbrot_set(-2.0f, 2.0f, -2.0f, 2.0f);

    {
        auto it = std::begin(v);

        for (auto i = 0_z; i < h; ++i, it += w) {
            std::copy(it, it + w, std::ostream_iterator<float>(std::cout, "\t"));
            std::cout << std::endl;
        }
    }

    return 0;
}
