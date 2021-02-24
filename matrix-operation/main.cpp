#include <iostream>
#include <vector>

#include "../util/util.hpp"

const std::size_t M = 1001;
const std::size_t N = 1002;
const std::size_t K = 1003;

inline auto matmul(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, std::vector<float>& matrix_c, std::size_t size_m, std::size_t size_n, std::size_t size_k) noexcept {
    for (auto i = 0_z; i < size_m; ++i) {
        for (auto j = 0_z; j < size_n; ++j) {
            matrix_c[i * size_n + j] = 0.0f;

            for (auto k = 0_z; k < size_k; ++k) {
                matrix_c[i * size_n + j] += matrix_a[i * size_k + k] * matrix_b[k * size_n + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    auto a = util::linspace(0.0f, 1.0f, M * K);
    auto b = util::linspace(0.0f, 1.0f, K * N);
    auto c = std::vector<float>(M * N, 0.0f);

    util::timeit([&]() {
        matmul(a, b, c, M, N, K);
    });

    matmul(a, b, c, M, N, K);

    std::cout << c[0        ] << std::endl;
    std::cout << c[1        ] << std::endl;
    std::cout << c[2        ] << std::endl;

    std::cout << c[M * N - 3] << std::endl;
    std::cout << c[M * N - 2] << std::endl;
    std::cout << c[M * N - 1] << std::endl;

    return 0;
}
