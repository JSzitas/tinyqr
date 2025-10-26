// Tiny QR solver, header only library
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (C) 2023- Juraj Szitas
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <benchmark/benchmark.h>

#include <iostream>

#include "../tinyqr.h"
#include "./utils.h"

/*
// The benchmark function
static void BM_MyFunction(benchmark::State& state) {
  // Setup: prepare data
  using scalar_t = float;
  const auto A = read_vec<scalar_t>("../benchmarking/A2.txt");
  const auto y = read_vec<scalar_t>("../benchmarking/y2.txt");
  auto myFunction = [&]() {
    auto coef = tinyqr::lm(A, y);
    return;
  };
  // Run: measure performance in a loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(A.data());
    benchmark::DoNotOptimize(y.data());
    myFunction();
    benchmark::ClobberMemory();
  }
}

// Register the function as a benchmark and specify a range for the argument
BENCHMARK(BM_MyFunction);

// Provide main() for Google Benchmark
BENCHMARK_MAIN();
*/

int main() {
  using scalar_t = float;
  /*const auto A = read_vec<scalar_t>("../benchmarking/A2.txt");
  const auto y = read_vec<scalar_t>("../benchmarking/y2.txt");
  auto f_lam = [&]() {
    auto coef = tinyqr::lm(A, y);
    return;
  };*/

  for (size_t i = 2; i < 512; i *= 2) {
    for (size_t j = (2 * i); j < 4096; j *= 2) {
      auto X = make_random_matrix<scalar_t>(j, i);
      const auto coefs = make_random_coefs<scalar_t>(i);
      auto y = make_y(X, coefs, j, i);
      auto f_lam = [&]() {
        auto coef = tinyqr::lm(X, y, static_cast<scalar_t>(10.));
        return;
      };
      std::cout << "n: " << j << " | p: " << i << " | ";
      custom_bench::benchmark<scalar_t>(f_lam, 100);
    }
  }
  return 0;
}
