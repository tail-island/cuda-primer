#include <iostream>
#include <random>

#include "../util/util.hpp"

inline auto is_in_circle(float x, float y) noexcept {
    return std::pow(x, 2) + std::pow(y, 2) <= 1.0f;
}

inline auto monte_carlo_pi(int n, unsigned int seed) noexcept {
    auto rng = std::mt19937(seed);
    auto distribution = std::uniform_real_distribution(0.0f, 1.0f);

    auto c = 0;

    for (auto i = 0; i < n; ++i) {
        if (is_in_circle(distribution(rng), distribution(rng))) {
            c++;
        }
    }

    return 4.0f * static_cast<float>(c) / static_cast<float>(n);
}

int main(int argc, char** argv) {
    util::timeit([]() {
        monte_carlo_pi(100'000'000, 0ul);
    });

    std::cout << monte_carlo_pi(100'000'000, 0u) << std::endl;

    return 0;
}
