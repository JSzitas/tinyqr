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

#include <iostream>

#include "./tinyqr.h"
#include "benchmarking/utils.h"

int main() {
  using scalar_t = float;
  using tinyqr::qr_decomposition;
  using tinyqr::internal::validate_qr;
  // you can validate easily using tinyqr::validate_qr - this is not done here
  // since I already did that elsewhere
  [&]() {
    const size_t n_iter = 100000;
    for (const auto i : {8, 16, 32, 64}) {  //{5, 8, 13, 32}) {
      for (const auto j : {256}) {
        std::vector<double> mean;
        StreamingMedian<scalar_t> median;
        std::cout << "n: " << j << " | p: " << i;
        auto X = make_random_matrix<float>(j, i, 0.0, 1.0);
        auto test =
            qr_decomposition<scalar_t>(X, j, i, static_cast<scalar_t>(1e-8));
        validate_qr<scalar_t, false>(X, test.Q, test.R, j, i);
        Stopwatch<false, std::chrono::duration<double, std::nano>> sw;
        for (size_t k = 0; k < n_iter; k++) {
          sw.reset();
          auto res =
              qr_decomposition<scalar_t>(X, j, i, static_cast<scalar_t>(1e-8));
          const auto timing = sw();
          median.push_back(timing);
          mean.push_back(timing);
        }
        scalar_t mean_ = 0.0;
        for (auto& val : mean) mean_ += val;
        mean_ /= n_iter;
        std::cout << " | median timing: " << median.value() * 1e-9
                  << " | mean timing: " << mean_ * 1e-9 << std::endl;
      }
    }
  }();
  return 0;
}
