#include <iostream>
#include <iterator>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

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

// inline std::tuple<thrust::host_vector<int>, std::size_t, std::size_t> mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size) {
//     auto cs_device = thrust::device_vector<thrust::complex<float>>(real_size * imag_size);
//     auto it = thrust::counting_iterator<std::size_t>(0);
//     thrust::transform(it, it + real_size * imag_size, std::begin(cs_device),
//         [=] __device__ (int i) {
//             const auto real_value = real_min + (real_max - real_min) / static_cast<float>(real_size - 1) * static_cast<float>(i % imag_size);
//             const auto imag_value = imag_min + (imag_max - imag_min) / static_cast<float>(imag_size - 1) * static_cast<float>(i / imag_size);

//             return thrust::complex(real_value, imag_value);
//         }
//     );

//     auto result_device = thrust::device_vector<int>(real_size * imag_size);
//     thrust::transform(std::begin(cs_device), std::end(cs_device), std::begin(result_device),
//         [=] __device__ (thrust::complex<float>& c) -> int {
//             return divergence_count(c);
//         }
//     );

//     return std::make_tuple(thrust::host_vector<int>(result_device), real_size, imag_size);
// }

inline std::tuple<thrust::host_vector<int>, std::size_t, std::size_t> mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size) {
    auto result_device = thrust::device_vector<int>(real_size * imag_size);
    auto it = thrust::counting_iterator<std::size_t>(0);
    thrust::transform(it, it + real_size * imag_size, std::begin(result_device),
        [=] __device__ (int i) -> int {
            return divergence_count([&]() {
                const auto real_value = real_min + (real_max - real_min) / static_cast<float>(real_size - 1) * static_cast<float>(i % imag_size);
                const auto imag_value = imag_min + (imag_max - imag_min) / static_cast<float>(imag_size - 1) * static_cast<float>(i / imag_size);

                return thrust::complex(real_value, imag_value);
            }());
        }
    );

    return std::make_tuple(thrust::host_vector<int>(result_device), real_size, imag_size);
}

inline auto mandelbrot_set(float real_min, float real_max, float imag_min, float imag_max) {
    auto [real_size, imag_size] = [&]() {
        auto real_diff = real_max - real_min;
        auto imag_diff = imag_max - imag_min;

        return std::make_tuple(
            static_cast<std::size_t>(1024 * std::min(real_diff / imag_diff, 1.0f)),
            static_cast<std::size_t>(1024 * std::min(imag_diff / real_diff, 1.0f))
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
