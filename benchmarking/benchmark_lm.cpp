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

#include "../tinyqr.h"
#include "./benchmark_utils.h"

int main() {
  /*
  const std::vector<double> A = {2.68, 8.86, 2.78, 0.09, 8.86, 50.93,
                                 3.78, 6.74, 2.78, 3.78, 8.95, 2.94,
                                 0.09, 6.74, 2.94, 45.46};
  // solving a linear system using QR
  std::vector<double> y = {-0.18, 0.56,  -9.7,  -3.21,
                           3.78,  16.94, -1.37, -51.32};
                           */
  using scalar_t = double;
  // coefficients for this should be 0.5 and -1.3, R's lm() gives 0.4938,
  // -1.1994
  const auto A = read_vec<scalar_t>("A2.txt");
  const auto y = read_vec<scalar_t>("y2.txt");
  auto coef = tinyqr::lm(A, y);
  for (const auto& coef_ : coef) std::cout << coef_ << ",";
  std::cout << '\n';

  const auto fun = [&]() {
    auto coef = tinyqr::lm(A, y);
    return;
  };
  benchmark<scalar_t>(fun);
  return 0;
}
