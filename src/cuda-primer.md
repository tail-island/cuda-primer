「この処理、もっと高速化して」と言われて「コンピューターを何百台も用意するとか、バカ高いライブラリを買うとか、金を湯水のように注ぎ込んで強力なクラウドを使うとかで、どうでしょうか？」と答えたら怒られました。

であれば、アルゴリズムの改良……はまず最初にやるべきで、で、もしすでにアルゴリズムを改良した後なのであれば、GPGPU（General-Purpose computing on Graphics Processing Units）なんてどうでしょうか？

本稿では、GPGPUの一つであるCUDAを使用して、プログラムを高速化する様々な方法を述べていきます。コードは[GitHub](https://github.com/tail-island/cuda-primer)にありますので、クローンしてみてください。

# 並列化してみよう

まず最初に理解していただきたいこと、それは、GPUは遅いということです。足し算や掛け算を普通にやらせたら、実は、GPUの速度はCPUの足元にも及びません。

試してみましょう。題材は私のようなおっさんプログラマの100%が喜ぶ[マンデルブロ集合](https://ja.wikipedia.org/wiki/%E3%83%9E%E3%83%B3%E3%83%87%E3%83%AB%E3%83%96%E3%83%AD%E9%9B%86%E5%90%88)です。まずは、普通にC++17で作ってみました。

~~~c++
#include <complex>
#include <iostream>
#include <iterator>
#include <tuple>
#include <vector>

#include "../util/util.hpp"

// マンデルブロの漸化式が何回で無限大に発散するか。発散しない場合は、0を返します。
inline auto divergence_count(const std::complex<float>& c) noexcept {
    auto z = std::complex(0.0f, 0.0f);

    for (auto i = 0; i < 100; ++i) {
        if (std::isinf(z.real())) {
            return i;
        }

        z = std::pow(z, 2) + c;
    }

    return 0;
}

// real_minからreal_max、imag_minからimag_maxの範囲の複素数平面をreal_size×imag_sizeに区切って、それぞれがマンデルブロ集合に含まれるか確認します。
inline auto mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size) {
    auto result = std::vector<int>(); result.reserve(real_size * imag_size);

    for (auto imag: util::linspace<float>(imag_min, imag_max, imag_size)) {
        for (auto real: util::linspace<float>(real_min, real_max, real_size)) {
            result.emplace_back(divergence_count(std::complex(real, imag)));
        }
    }

    return std::make_tuple(result, real_size, imag_size);
}

// 適切なreal_sizeとimag_sizeを作成して、mandelbrot_set()を呼び出します。
inline auto mandelbrot_set(float real_min, float real_max, float imag_min, float imag_max) {
    auto [real_size, imag_size] = [&]() {  // ラムダ式を使うと変数のスコープを短くできて便利。構造化束縛を使うと複数の値をリターンできて便利。
        auto real_diff = real_max - real_min;
        auto imag_diff = imag_max - imag_min;

        return std::make_tuple(
            static_cast<std::size_t>(1024 * std::min(real_diff / imag_diff, 1.0f)),
            static_cast<std::size_t>(1024 * std::min(imag_diff / real_diff, 1.0f))
        );
    }();

    return mandelbrot_set(real_min, real_max, real_size, imag_min, imag_max, imag_size);
}

// メイン・ルーチン。
int main(int argc, char** argv) {
    // 時間計速。util::timeitは自作の処理時間計測ルーチン。
    util::timeit([&]() {
        mandelbrot_set(-2.0f, 2.0f, -2.0f, 2.0f);
    });

    // プログラムが正しいかを確認するため、マンデルブロ集合を出力します。
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
~~~

動かしてみましょう。出力を画像にして表示する処理は、Pythonで作成しました。

~~~shell
$ make && ./mandelbrot | python plot.py
~~~

![マンデルブロ集合](https://github.com/tail-island/cuda-primer/blob/gh-pages/src/image/mandelbrot-set.png)

はい。きれいにマンデルブロ集合が表示されました。マンデルブロ集合生成にかかった時間は、
