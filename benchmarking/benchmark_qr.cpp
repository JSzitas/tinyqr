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

#include <iostream>  // NOLINT

#include "../tinyqr.h"
#include "./utils.h"

int main() {
  const std::vector<double> A = {2.68, 8.86, 2.78, 0.09, 8.86, 50.93,
                                 3.78, 6.74, 2.78, 3.78, 8.95, 2.94,
                                 0.09, 6.74, 2.94, 45.46};

  using scalar_t = double;
  auto benchmark = Benchmarker<scalar_t>(1000);
  /*
    size_t i = 10, j = 1000;
    auto X = make_random_matrix<scalar_t>(j, i);
    std::function<void()> v1 = [&]() { tinyqr::qr_decomposition(X, j, i); };
    std::function<void()> v2 = [&]() { tinyqr::qr_decomposition2(X, j, i); };
    std::cout << "n: " << j << " | p: " << i << std::endl;
    benchmark_versions(v1,v2);
    */

  for (size_t i = 2; i < 128; i *= 2) {
    for (size_t j = i; j < 2048; j *= 2) {
      auto X = make_random_matrix<scalar_t>(j, i);
      std::function<void()> v1 = [&]() { tinyqr::qr_decomposition(X, j, i); };
      std::function<void()> v2 = [&]() { tinyqr::qr_decomposition2(X, j, i); };
      std::cout << "n: " << j << " | p: " << i << std::endl;
      benchmark(v1, v2);
    }
  }

  benchmark.report();
  return 0;
}
