「この処理、もっと高速化して」と言われて「コンピューターを何百台も用意するとか、バカ高いライブラリを買うとか、金を湯水のように注ぎ込んで強力なクラウドを借りるとかでどうでしょうか？」と答えたら怒られました。

であれば、アルゴリズムの改良……はまず最初にやるべきで、で、もしすでにアルゴリズムを改良した後なのであれば、GPGPU（General-Purpose computing on Graphics Processing Units）なんてどうでしょうか？

本稿では、GPGPUの一つであるCUDAを使用して、プログラムを高速化する様々な方法を述べていきます。コードは[GitHub](https://github.com/tail-island/cuda-primer)にありますので、クローンしてみてください。

# GPUは遅い！

まず最初に理解していただきたいこと、それは、GPUは遅いということです。足し算や掛け算を普通にやらせたら、実は、GPUの速度はCPUの足元にも及びません。

試してみましょう。題材は私のようなおっさんプログラマの100%が喜ぶ[マンデルブロ集合](https://ja.wikipedia.org/wiki/%E3%83%9E%E3%83%B3%E3%83%87%E3%83%AB%E3%83%96%E3%83%AD%E9%9B%86%E5%90%88)です。まずは、[普通にC++17](https://github.com/tail-island/cuda-primer/tree/main/mandelbrot)で作ってみました。

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
inline auto mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size) noexcept {
    auto result = std::vector<int>(); result.reserve(real_size * imag_size);

    for (auto imag: util::linspace<float>(imag_min, imag_max, imag_size)) {
        for (auto real: util::linspace<float>(real_min, real_max, real_size)) {
            result.emplace_back(divergence_count(std::complex(real, imag)));
        }
    }

    return std::make_tuple(result, real_size, imag_size);
}

// 適切なreal_sizeとimag_sizeを作成して、mandelbrot_set()を呼び出します。
inline auto mandelbrot_set(float real_min, float real_max, float imag_min, float imag_max) noexcept {
    auto [real_size, imag_size] = [&]() {  // ラムダ式を使うと変数のスコープを短くできて便利。構造化束縛を使うと複数の値をリターンできて便利。
        auto real_diff = real_max - real_min;
        auto imag_diff = imag_max - imag_min;

        return std::make_tuple(
            static_cast<std::size_t>(1024 * std::min(real_diff / imag_diff, 1.0f)),
            static_cast<std::size_t>(1024 * std::min(imag_diff / real_diff, 1.0f))
        );
    }();

    return mandelbrot_set(real_min, real_max, real_size, imag_min, imag_max, imag_size);で、
}

// メイン・ルーチン。
int main(int argc, char** argv) {
    // 時間計速。util::timeitは自作の処理時間計測ルーチン。./util/util.hppを参照してください。
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

![マンデルブロ集合](https://raw.githubusercontent.com/tail-island/cuda-primer/gh-pages/src/image/mandelbrot-set.png)

はい。きれいにマンデルブロ集合が表示されましたのでプログラムはこれで正しそう。で、C++17でマンデルブロ集合生成にかかった時間は、型落ちのラップトップPCで0.112794秒でした。

上のプログラムの`mandelbrot_set()`（2つある内の上の方）を[CUDAでGPU上で実行](https://github.com/tail-island/cuda-primer/tree/main/mandelbrot-cuda-ng)させてみます。コードは[こんな感じ](https://github.com/tail-island/cuda-primer/blob/main/mandelbrot-cuda-ng/main.cu)。

~~~c++
#include <complex>
#include <iostream>
#include <iterator>
#include <tuple>
#include <vector>

#include <thrust/complex.h>

#include "../util/cudautil.hpp"
#include "../util/util.hpp"

__device__  // GPU上で実行される関数には__device__を付けてください。
inline auto divergence_count(const thrust::complex<float>& c) noexcept {
    auto z = thrust::complex(0.0f, 0.0f);  // __device__関数ではライブラリの呼び出しに制限があって、std::complexが使えません。Thrustのcomplexで代用しました。

    for (auto i = 0; i < 100; ++i) {
        if (std::isinf(z.real())) {
            return i;
        }

        z = z * z + c;  // thrust::complexではstd::pow()を使用できなかったので、z * zに書き換えました。
    }

    return 0;
}

__device__
inline auto linspace(float start, float stop, std::size_t size) noexcept {  // util::linspace()は__device__ではないので、__device__バージョンを作成しました。
    auto result = new float[size];  // std::vectorは使用できなかったので、生ポインターで。

    auto delta = (stop - start) / static_cast<float>(size - 1);

    for (auto i = static_cast<std::size_t>(0); i < size; ++i) {
        result[i] = start + delta * i;
    }

    return result;
}

__global__  // GPU上で実行され、かつ、CPUから呼び出される関数には__global__をつけてください。
void mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size, int* result) {  // __global__の戻り値はvoidです。また、noexceptも付加できません。
    auto imags = linspace(imag_min, imag_max, imag_size);  // あとでdeleteしなければならないので、変数に保存しておきます。
    auto reals = linspace(real_min, real_max, real_size);

    for (auto i = 0; i < imag_size; ++i) {  // ポインターだと拡張forが使えないので、普通のforを使います。
        for (auto j = 0; j < real_size; ++j) {
            result[i * real_size + j] = divergence_count(thrust::complex(reals[j], imags[i]));
        }
    }

    delete reals;  // linspaceはnewしたポインターを返すので、自前でdeleteします。
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

    int* result_device;  // GPUのメモリとCPUのメモリは異なるので、GPU用のメモリを用意します。
    cuda_check(cudaMalloc(&result_device, sizeof(int) * real_size * imag_size));  // GPUのメモリの確保はcudaMallocで。エラー・チェックは戻り値でやるので、cudautil::cuda_check()という関数を自作しました。

    mandelbrot_set<<<1, 1>>>(real_min, real_max, real_size, imag_min, imag_max, imag_size, result_device);  // __global__の呼び出しには、<<<block, grid>>>で並列度を指定します。今回は1並列（並列実行しない）でやります。
    cuda_check(cudaGetLastError());  // __global__の呼び出しでエラーが発生してないかのチェック。

    cuda_check(cudaMemcpy(result.data(), result_device, sizeof(int) * real_size * imag_size, cudaMemcpyDeviceToHost));  // GPUのメモリーからCPUのメモリーに結果をコピーします。

    cuda_check(cudaFree(result_device));  // GPUのメモリ開放はcudaFreeでやります。

    return std::make_tuple(result, real_size, imag_size);
}

int main(int argc, char** argv) {
    first_cudaMalloc_is_too_slow();  // 一回目のcudaMallocは遅いので、タイム計測に影響しないように一回cudaMallocしておきます。
    cuda_check(cudaDeviceSynchronize());  // CUDAの関数呼び出しは非同期なので、タイム計測に影響しないようにGPU上の処理が終わるまで待ちます。

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
~~~

あぁいろいろと面倒くせぇ……のですけど、順を追っていけば別に難しくはありません。

まずは、2つある内の下の方の`mandelbrot_set()`を見てみてください。`cudaMalloc()`したり`cudaMemcpy()`したり`cudaFree()`したりしています。なぜこんな面倒なことをしているかというと、*GPUのメモリとCPUのメモリは異なる*ためです。

![GPUのメモリとCPUのメモリは異なる]()

だから、`std::vector`でCPUのメモリに場所を確保した上で`cudaMalloc()`でGPUのメモリを確保して、で、GPUのメモリ上に生成されたマンデルブロ集合の情報を`cudaMemcpy()`でCPUのメモリに転送してあげなければなりません。また、確保したメモリは開放しなければならないので、`cudaFree()`を呼ぶのも忘れてはなりません（実は後述するThrustを使うと少し楽ができるのですけど）。あ、関数の引数は、普通に渡しても大丈夫です。そうじゃないと、GPUのメモリ・アドレスを渡せないですしね。

あと、CUDAの関数は例外を使用していないので、戻り値でエラー・チェックしなければなりません。あぁ面倒くせぇけど、これはエラー・チェック用の関数を作成すれば少し楽になります。今回は`cuda_check()`という関数を作成しました。

~~~c++
#include <cstdlib>

#include <cuda.h>

inline auto cuda_check(const cudaError_t& error) noexcept {
    if (error == cudaSuccess) {
        return;
    }

    std::cerr << cudaGetErrorString(error) << std::endl;
    std::exit(1);
}
~~~

さて、GPUのメモリを確保できたので、次はGPU上で実行する処理を書きます。その処理が、２つあるうちの上の方の`mandelbrot_set()`です。こちらを見てみると、関数定義の前に`__global__`って書いてあります。この`__global__`を付加してあげると、CPUから呼び出せる、GPUで実行される関数になるんですね。で、`__global__`な関数は、戻り値を使用できない（`void`でなければならない）、`inline`や`noexcept`等を使用できない、可変引数や再帰処理や性的変数を使ってはダメ等の制約はあるのですが、それらを除けば、普通にC++のコードを書けます。ただね、原則的に、CPU向けの関数を呼び出すことができないんですよ……。たとえば`std::complex`も`std::vector`も使えません。というかね、自作した関数すら呼び出せません。自作した関数を`__global__`から呼び出したいなら、上のコードの`divergence_count()`や`linspace()`みたいに関数に`__device__`をつけてあげないとダメ。で、その中ではやっぱり標準ライブラリを呼び出せません（`divergence_count()`で使用している`std::isinf()`のような算術関数は例外的に使用可）。というわけで、標準ライブラリを使わない形にいろいろ書き換えました。あ、`std::complex`の代わりに使用している`thrust::complex`は、後述するNVIDIAのThrustライブラリが提供する複素数型です。`new`とか`delete`なんて久しぶりに使いましたよ。あぁ面倒くせぇ。

でも、この標準ライブラリが使えない問題は、CUDA 10.2で追加されたlibcu++によって、今後は改善されていくと思われます。libcu++はCUDA版の標準ライブラリで、今はまだあまりカバー範囲が広くないのですけど、そのうち標準ライブラリの多くがサポートされるようになって、今より普通にC++プログラミングできるようになるんじゃないかな。2021年3月現在では、サポートする範囲が狭いのであまり使えないですけどね。

あとは、`__global__`関数を呼び出すだけ。`___global___`を呼び出すには上のコードのように`mandelbrot_set<<<1, 1>>>(real_min, real_max, real_size, imag_min, imag_max, imag_size, result_device);`みたいにします。普通のC++と違うのは`<<<1, 1>>>`の部分で、ここはどんなふうに並列化するかを表現する部分なのですけど、今回は`1, 1`で1並列で実行します。あと、`__global__`や`<<<1, 1>>>`は普通のC++コンパイラでは処理できないので、CUDA用のコンパイラの`nvcc`でコンパイルしなければならなくて、あと拡張子は*.cuが標準になります。詳しくは[makefile](https://github.com/tail-island/cuda-primer/blob/main/mandelbrot-cuda-ng/makefile)を参照してください。あぁ本当にいろいろと面倒くせぇ！

というわけで、CUDAプログラミングのやり方をまとめると、以下になります。

* ソース・コードの拡張子は*.cu。nvccでコンパイルする。
* CPUから呼び出すGPUで実行する関数には、`__global__`を付ける。戻り値は`void`。
* `__global__`関数から呼び出す関数は、`__device__`を付ける（今回は使用していませんが、CPUでも使用したいなら`__device__ __host__`としてください）。
* `__global__`関数や`__device__`関数からは標準ライブラリを呼べない。標準ライブラリなしで頑張る。
* `__global__`関数を呼び出すときは、`<<<grid, block>>>`をつけて呼び出す。
* CPUに渡したり、CPUから受け取ったりするデータの内、引数で表現できないものは`cudaMalloc()`で確保し、`cudaFree()`で開放する。
* CPUとの間のメモリのコピーは、`cudaMemcpy()`で実施する。
* エラー・チェックは戻り値のチェックで実施。
* 標準ライブラリは、算術関数以外は使用できない。
* とにかくいろいろ面倒くせぇ。

とまぁ、これだけ苦労してCUDA化した結果を見てみると、私の型落ちのGeForce GTX 1080 Tiでの`mandelbrot_set()`の実行にかかった時間は……0.652916秒でした。うん、CPUの5.79倍くらい*遅い*！　というわけで、いいですか、GPUは遅いんです……。

# GPUは早い！

じゃあなんでこんな苦労してわざわざGPGPUするのかといえば、「GPUは早い」からなんです……と言われても、遅い証拠を見せられた後なのでなんだかわからないですよね。一体なんなのか、CPUをスポーツ・カーに、GPUをバスに例えて説明させてください。

CPUはスポーツカーなので速度はスゴイのですけど、少人数しか乗れません。GPUはバスなので速度はそこそこなのですけど、大人数が乗れます。さて、大人数をA地点からB地点まで運ぶという目的の場合は、どちらが早く終了するでしょうか？　そう、スポーツ・カーでピストン輸送するよりも、バスに大人数を載せて一回で運んだ方が処理に必要な時間は少ないというわけ。

コンピューター処理で「大人数をA地点からB地点まで運ぶ」に相当するのは、同じ処理（A地点からB地点まで運ぶ）を何回も繰り返す（大人数）場合になります。ピストン輸送に相当するループ処理の内側を、ループ無しで並列で同時に実行する。これで所要時間が短くなるというわけ。先程までのコードでは、CPUもGPUも乗客を1人しか載せずにピストン輸送していたんですな。それではバスの良さがでなくても当然。

というわけで、同時に何人も乗せる（処理を並列で実行する）ようにプログラムを修正してみましょう。考えてみると、今回のコードでは`divergence_count()`は1,024×1,024で100万回とちょっと呼ばれます。で、*-2-2i*の場合の漸化式が発散するかと*2+2i*の場合の漸化式が発散するかは無関係ですから、並列での実行が可能。だったら、CUDAで並列で実行しちゃえばよい。修正した結果は、こんな感じになります。

~~~c++
#include <iostream>
#include <iterator>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <thrust/complex.h>

#include "../util/cudautil.hpp"
#include "../util/util.hpp"

// mandelbrot-cuda-ngと同じ。標準ライブラリを使用しない形に書き換えただけです。
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

// この関数をGPU上で並列で実行します。
__global__
void mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size, int* result) {
    const auto i = blockDim.x * blockIdx.x + threadIdx.x;  // blockDimやblockIdx、threadIdxはCUDAが提供する変数。スレッドのIDを表現します。詳細は本文で解説。

    const auto real_value = real_min + (real_max - real_min) / static_cast<float>(real_size - 1) * static_cast<float>(threadIdx.x);  // linspace()の処理を移植。スレッドのIDから、対象となる複素数を計算します。
    const auto imag_value = imag_min + (imag_max - imag_min) / static_cast<float>(imag_size - 1) * static_cast<float>(blockIdx.x);   // 同上。threadIdxやblockIdxについては、本文で解説します。

    result[i] = divergence_count(thrust::complex(real_value, imag_value));
}

// GPU上で実行する関数を、この関数から呼び出します。
inline auto mandelbrot_set(float real_min, float real_max, std::size_t real_size, float imag_min, float imag_max, std::size_t imag_size) noexcept {
    auto result = std::vector<int>(real_size * imag_size);

    int* result_device;
    cuda_check(cudaMalloc(&result_device, sizeof(int) * real_size * imag_size));

    // imag_size×real_size個の並列で、mandelbrot_setを実行します。
    mandelbrot_set<<<imag_size, real_size>>>(real_min, real_max, real_size, imag_min, imag_max, imag_size, result_device);
    cuda_check(cudaGetLastError());

    cuda_check(cudaMemcpy(result.data(), result_device, sizeof(int) * real_size * imag_size, cudaMemcpyDeviceToHost));

    cuda_check(cudaFree(result_device));

    return std::make_tuple(result, real_size, imag_size);
}

// mandelbrotと同じ。
inline auto mandelbrot_set(float real_min, float real_max, float imag_min, float imag_max) noexcept {
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

// mandelbrot-cuda-ngと同じ。
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
~~~

今回の本質的な変更は、`<<<1, 1>>>`から`<<<imag_size, real_size>>>`への変更です。このように書くだけで、`imag_size`×`real_size`分の100万ちょっとのスレッドが生成され、並行で実行されます。100万回のピストン輸送ではなく一回の輸送になるのだから、そりゃ早く済んで当たり前ですな（GPU内のコアの数次第ではありますけど）。

あと、呼び出されるスレッドの方では、自分がどの要素を処理すればよいのかのかを知りたい。CUDAではない普通のスレッド・プログラミングなら引数で情報を渡すのですけど、CUDAではスレッドは同期して実行する（複数のスレッドが同じマシン語を実行します。たとえば、if文で条件分岐するような場合は、条件が偽となったスレッドは、他のスレッドで真となった場合の処理が終わるまで何もしないで待つ）ため、一気にスレッドを起動しなければならなくて、だからスレッド一つ一つに引数としてパラメーターを渡すことができません。ではどうするのかというと、スレッドに一意になるIDを割り振ることで対応します。スレッド側からのIDの取得は、`threadIdx.x`のような暗黙の変数で取得します。

……で、ここまでを読んで、`<<<1, 1>>>`のように2つ値を指定していたのは何でだ（スレッドの数を指定するなら一つで良いはず）とか、`threadIdx.x`の`x`は何だ（`<<<1, 1>>>`の１つ目がyで2つ目がxなのだとしたら、コード中の`blockIdx.x`が意味不明になる）とか、そのような疑問を感じたことと思います。この疑問への回答を得るには、CUDAのスレッドの実行モデルを理解しなければなりません。

図

上の図のように、CUDAでは、グリッドの中にブロックがあって、このブロックの中にスレッドがあります。`<<<1, 1>>>`の１つ目はグリッドの中に何個のブロックを作るのかの指定で、2つ目はブロックの中に何個のスレッドを作るのかの指定なんです。で、ブロック中に作成できるスレッドの数には最大値の制限があって（私が使用しているGeForce GTX 1080 Tiでは1,024スレッドまで）、あと、ブロック中のスレッドの数が32の場合に効率が良かったりします。だから、例えば2次元表を更新するような場合であっても、y方向をブロック、x方向をスレッドとするわけにはいきません。だから、CUDAでは`dim3`という3次元上の位置を表現できるクラスを用意してくれていて、それを使うことを推奨しています。今回の場合だと、CUDAの推奨に従った場合は、`__global__`関数を呼び出す場合は以下のようになります。

~~~c++
~~~

で、`dim3`には`x`と`y`、`z`の属性があるのですけど、
