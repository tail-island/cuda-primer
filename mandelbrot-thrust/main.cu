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

struct divergence_count {
    __device__
    inline int operator()(const thrust::complex<float>& c) const noexcept {
        auto z = thrust::complex(0.0f, 0.0f);

        for (auto i = 0; i < 100; ++i) {
            if (std::isinf(z.real())) {
                return i;
            }

            z = z * z + c;
        }

        return 0;
    }
};

__global__
void initialize_complexes(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size, thrust::complex<float>* result) {
    const auto i = blockDim.x * blockIdx.x + threadIdx.x;

    const auto real_value = real_min + (real_max - real_min) / static_cast<float>(real_size - 1) * static_cast<float>(threadIdx.x);
    const auto imag_value = imag_min + (imag_max - imag_min) / static_cast<float>(imag_size - 1) * static_cast<float>(blockIdx.x);

    result[i] = thrust::complex(real_value, imag_value);
}

inline auto mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size) {
    // auto cs_device = thrust::device_vector<thrust::complex<float>>(real_size * imag_size);

    // initialize_complexes<<<imag_size, real_size>>>(real_min, real_max, real_size, imag_min, imag_max, imag_size, cs_device.data().get());
    // cuda_check(cudaGetLastError());

    auto cs_device = thrust::device_vector<thrust::complex<float>>(
        [&]() {
            auto result = thrust::host_vector<thrust::complex<float>>(); result.reserve(real_size * imag_size);

            for (auto imag: util::linspace<float>(imag_min, imag_max, imag_size)) {
                for (auto real: util::linspace<float>(real_min, real_max, real_size)) {
                    result.push_back(thrust::complex(real, imag));
                }
            }

            return result;
        }()
    );

    auto result_device = thrust::device_vector<int>(real_size * imag_size);
    thrust::transform(std::begin(cs_device), std::end(cs_device), std::begin(result_device), divergence_count());

    auto result = thrust::host_vector<int>(result_device);

    return std::make_tuple(result, real_size, imag_size);
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
