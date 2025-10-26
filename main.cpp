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
#include <vector>

#include "./tinyqr.h"

template <typename T>
[[maybe_unused]] void print_eigendecomposition_result(T &result,
                                                      const size_t n = 4) {
  std::cout << "Eigenvals: \n";
  for (auto &eigval : result.eigenvals) {
    std::cout << eigval << ",";
  }
  std::cout << std::endl;
  std::cout << "Eigenvecs: \n";
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      std::cout << result.eigenvecs[j * 4 + i] << ',';
    }
    std::cout << '\n';
  }
  std::cout << std::endl;
}
[[maybe_unused]] void print_vec(const std::vector<double> &x) {
  for (const auto &val : x) {
    std::cout << val << ",";
  }
  std::cout << "\n";
}
[[maybe_unused]] void print_square_mat(const std::vector<double> &x,
                                       const size_t n = 4) {
  std::cout << "\n";
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      std::cout << x[j * n + i] << ',';
    }
    std::cout << '\n';
  }
  std::cout << std::endl;
}

void print_rec_mat(const std::vector<double> &x, const size_t n = 4,
                   const size_t p = 4) {
  std::cout << "\n";
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < p; j++) {
      std::cout << x[j * n + i] << ',';
    }
    std::cout << '\n';
  }
  std::cout << std::endl;
}

void print_QR_decomposition(const tinyqr::QR<double> &qr, const size_t n,
                            const size_t p) {
  std::cout << "Q: " << std::endl;
  print_rec_mat(qr.Q, n, p);
  std::cout << "\n";
  std::cout << "R: " << std::endl;
  print_rec_mat(qr.R, p, p);
  std::cout << "\n";
}

int main() {
  // only works for symmetric matrices - which is my application (covariance
  // matrices)
  const std::vector<double> A = {2.68, 8.86, 2.78, 0.09, 8.86, 50.93,
                                 3.78, 6.74, 2.78, 3.78, 8.95, 2.94,
                                 0.09, 6.74, 2.94, 45.46};
  auto test = [&](const std::vector<double> &X, const size_t n,
                  const size_t p) {
    tinyqr::QR<double> QR_res = tinyqr::qr_decomposition(X.data(), n, p);
    print_QR_decomposition(QR_res, n, p);
    tinyqr::internal::validate_qr(X, QR_res.Q, QR_res.R, n, p);
  };
  // test(A, 4, 4);
  // test(A, 8, 2);

  const std::vector<double> A_ridge = {-0.39703613471675,
                                       0.490489932672572,
                                       -0.691290797580314,
                                       -0.421940489713308,
                                       -0.108034714877868,
                                       -1.57605913952942,
                                       -0.201548263825867,
                                       0.118711935863356,
                                       15,
                                       0,
                                       0,
                                       0,
                                       0.602853200120877,
                                       -0.385529099242147,
                                       0.30343740093931,
                                       -0.997514333841874,
                                       -1.38056788587163,
                                       -2.21734342228736,
                                       1.40181838861434,
                                       -0.696457647273841,
                                       0,
                                       15,
                                       0,
                                       0,
                                       1.52894909945742,
                                       1.86075470296447,
                                       -0.0934468327817949,
                                       1.66531245583755,
                                       0.601620454037474,
                                       0.672762033312901,
                                       0.11529310781543,
                                       1.43979533919943,
                                       0,
                                       0,
                                       15,
                                       0,
                                       -1.98600952388699,
                                       0.354056942466987,
                                       -0.0839134931014958,
                                       -1.33018909588546,
                                       -1.24932496022576,
                                       -0.756637521718698,
                                       -0.2526215851156,
                                       2.17902058547342,
                                       0,
                                       0,
                                       0,
                                       15};

  // solving a linear system using QR
  const std::vector<double> y_ridge = {1.20436142451793,
                                       1.56704905625387,
                                       -0.998940073008348,
                                       3.78356491810657,
                                       4.18262883210162,
                                       4.46453958673887,
                                       -2.61217924951138,
                                       -0.0431666772150563,
                                       0.,
                                       0.,
                                       0.,
                                       0.};
  const auto coef = tinyqr::lm(A_ridge, y_ridge);
  std::cout << "Linear system solution coefficients: ";
  for (const auto &coef_ : coef) std::cout << coef_ << ",";
  std::cout << "\n(against expectation of 0.384603, -0.986914)\n\n";

  const std::vector<double> A_ = {
      -0.39703613471675,   0.490489932672572,  -0.691290797580314,
      -0.421940489713308,  -0.108034714877868, -1.57605913952942,
      -0.201548263825867,  0.118711935863356,  0.602853200120877,
      -0.385529099242147,  0.30343740093931,   -0.997514333841874,
      -1.38056788587163,   -2.21734342228736,  1.40181838861434,
      -0.696457647273841,  1.52894909945742,   1.86075470296447,
      -0.0934468327817949, 1.66531245583755,   0.601620454037474,
      0.672762033312901,   0.11529310781543,   1.43979533919943,
      -1.98600952388699,   0.354056942466987,  -0.0839134931014958,
      -1.33018909588546,   -1.24932496022576,  -0.756637521718698,
      -0.2526215851156,    2.17902058547342};

  // solving a linear system using QR
  const std::vector<double> y_ = {1.20436142451793,   1.56704905625387,
                                  -0.998940073008348, 3.78356491810657,
                                  4.18262883210162,   4.46453958673887,
                                  -2.61217924951138,  -0.0431666772150563};
  const auto coef_ = tinyqr::lm(A_, y_, 15.);
  std::cout << "Linear system solution coefficients: ";
  for (const auto &coef_2 : coef_) std::cout << coef_2 << ",";
  std::cout << "\n(against expectation of 0.384603, -0.986914)\n\n";

  /*
  // solving a linear system using QR
  std::vector<double> y = {-0.18, 0.56,  -9.7,  -3.21,
                           3.78,  16.94, -1.37, -51.32};
  // coefficients for this should be 0.5 and -1.3, R's lm() gives 0.4938,
  // -1.1994
  auto coef = tinyqr::lm(A, y);
  std::cout << "Linear system solution coefficients: ";
  for (auto coef_ : coef) std::cout << coef_ << ",";
  std::cout << "\n(against expectation of 0.4938, -1.1994)\n\n";
  // QR Algorithm
  const auto res = tinyqr::qr_algorithm<double>(A, 25);
  print_eigendecomposition_result(res);
  auto solver = tinyqr::QRSolver<double>(4);
  // run solver
  solver.solve(A, 25);
  std::cout << "Eigenvals: \n";
  print_vec(solver.eigenvalues());
  std::cout << "Eigenvecs:";
  print_square_mat(solver.eigenvectors());
  */
  return 0;
}
