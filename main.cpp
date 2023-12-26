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

int main() {
  // only works for symmetric matrices - which is my application (covariance
  // matrices)
  const std::vector<double> A = {2.68, 8.86, 2.78, 0.09, 8.86, 50.93,
                                 3.78, 6.74, 2.78, 3.78, 8.95, 2.94,
                                 0.09, 6.74, 2.94, 45.46};
  const auto res = tinyqr::qr_algorithm(A);
  std::cout << "Eigenvals: \n";
  for (auto &eigval : res.eigenvals) {
    std::cout << eigval << ",";
  }
  std::cout << std::endl;
  std::cout << "Eigenvecs: \n";
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      std::cout << res.eigenvecs[j * 4 + i] << ',';
    }
    std::cout << '\n';
  }
  std::cout << std::endl;
  return 0;
}
