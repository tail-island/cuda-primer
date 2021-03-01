#include <iostream>
#include <random>

#include <curand.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "../util/cudautil.hpp"
#include "../util/util.hpp"

__device__
inline auto is_in_circle(float x, float y) noexcept {
    return std::pow(x, 2) + std::pow(y, 2) <= 1.0f;
}

inline float monte_carlo_pi(int n, unsigned long seed) noexcept {
    auto rng = curandGenerator_t();
    // curand_check(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MT19937));
    curand_check(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MTGP32));
    curand_check(curandSetPseudoRandomGeneratorSeed(rng, seed));

    auto xs = thrust::device_vector<float>(n);
    curand_check(curandGenerateUniform(rng, xs.data().get(), n));

    auto ys = thrust::device_vector<float>(n);
    curand_check(curandGenerateUniform(rng, ys.data().get(), n));

    auto it = thrust::make_zip_iterator(thrust::make_tuple(std::begin(xs), std::begin(ys)));
    auto c  = thrust::count_if(it, it + n,
        [=] __device__ (const thrust::tuple<float, float>& p) {
            return is_in_circle(thrust::get<0>(p), thrust::get<1>(p));
        }
    );

    return 4.0f * static_cast<float>(c) / static_cast<float>(n);
}

int main(int argc, char** argv) {
    first_cudaMalloc_is_too_slow();

    cuda_check(cudaDeviceSynchronize());

    util::timeit([]() {
        monte_carlo_pi(100'000'000, 0ul);
    });

    std::cout << monte_carlo_pi(100'000'000, 0ul) << std::endl;

    return 0;
}
