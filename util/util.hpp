#include <chrono>
#include <type_traits>

inline std::size_t operator "" _z(unsigned long long x) {
    return x;
}

namespace util {
    template <typename T>
    inline auto linspace(T start, T stop, std::size_t size) noexcept {
        auto result = std::vector<T>(size);

        const auto delta = (stop - start) / static_cast<T>(size - 1);

        for (auto i = static_cast<std::size_t>(0); i < size; ++i) {
            result[i] = start + delta * static_cast<T>(i);
        }

        return result;
    }

    template <typename T>
    inline auto timeit(T functor) noexcept {
        auto t = 0_z;

        for (auto i = 0; i < 10; ++i) {
            const auto start = std::chrono::system_clock::now();

            functor();

            t += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
        }

        std::cerr << (static_cast<float>(t) / 1000000.0f / 10.0f) << std::endl;
    }
}
