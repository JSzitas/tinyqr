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

#ifndef TINYQR_H_
#define TINYQR_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace tinyqr::internal {
template <typename scalar_t>
std::tuple<scalar_t, scalar_t> givens_rotation(scalar_t a, scalar_t b) {
  if (std::abs(b) <= std::numeric_limits<scalar_t>::min()) {
    return std::make_tuple(1.0, 0.0);
  } else {
    if (std::abs(b) > std::abs(a)) {
      const scalar_t r = a / b;
      const scalar_t s = 1.0 / sqrt(1.0 + std::pow(r, 2));
      return std::make_tuple(s * r, s);
    } else {
      const scalar_t r = b / a;
      const scalar_t c = 1.0 / sqrt(1.0 + std::pow(r, 2));
      return std::make_tuple(c, c * r);
    }
  }
}
template <typename scalar_t>
[[maybe_unused]] inline void A_matmul_B_to_C(std::vector<scalar_t> &A,
                                             std::vector<scalar_t> &B,
                                             std::vector<scalar_t> &C,
                                             const size_t ncol) {
  for (size_t k = 0; k < ncol; k++) {
    for (size_t l = 0; l < ncol; l++) {
      scalar_t accumulator = 0.0;
      for (size_t m = 0; m < ncol; m++) {
        accumulator += A[m * ncol + k] * B[l * ncol + m];
      }
      C[l * ncol + k] = accumulator;
    }
  }
}
template <typename scalar_t>
inline void A_matmul_tB_to_C(std::vector<scalar_t> &A, std::vector<scalar_t> &B,
                             std::vector<scalar_t> &C, const size_t ncol) {
  for (size_t k = 0; k < ncol; k++) {
    for (size_t l = 0; l < ncol; l++) {
      scalar_t accumulator = 0.0;
      for (size_t m = 0; m < ncol; m++) {
        accumulator += A[m * ncol + k] * B[m * ncol + l];
      }
      C[l * ncol + k] = accumulator;
    }
  }
}
// this is the implementation of QR decomposition - this does not get exposed,
// only the nice(r) facades do
template <typename scalar_t, const bool cleanup = false>
void qr_impl(std::vector<scalar_t> &Q, std::vector<scalar_t> &R, const size_t n,
             const size_t p, const scalar_t tol) {
  for (size_t j = 0; j < p; j++) {
    for (size_t i = n - 1; i > j; --i) {
      // using tuples and structured bindings should make this fairly ok
      // performance wise
      const auto [c, s] = givens_rotation(R[(j * n) + (i - 1)], R[j * n + i]);
      // you can make the matrix multiplication implicit, as the givens rotation
      // only impacts a moving 2x2 block
      for (size_t k = 0; k < p; ++k) {
        // first do G'R - keep in mind this is transposed
        const size_t upper = k * n + (i - 1);
        const size_t lower = k * n + i;
        const scalar_t temp_1 = R[upper];
        const scalar_t temp_2 = R[lower];
        // carry out the multiplies on required elements
        R[upper] = c * temp_1 + s * temp_2;
        R[lower] = -s * temp_1 + c * temp_2;
      }
      for (size_t k = 0; k < n; k++) {
        // QG - note that this is not transposed
        const size_t upper = k * n + i;
        const size_t lower = k * n + i - 1;
        const scalar_t temp_1 = Q[upper];
        const scalar_t temp_2 = Q[lower];
        // again, NOT transposed, so s and -s are flipped
        Q[upper] = c * temp_1 - s * temp_2;
        Q[lower] = s * temp_1 + c * temp_2;
      }
    }
  }
  // clean up R - particularly under the diagonal - only useful if you are
  // interested in the actual decomposition
  if constexpr (cleanup) {
    for (auto &val : R) {
      val = std::abs(val) < tol ? 0.0 : val;
    }
  }
}
// this computes the Gram matrix (here dubbed 'grammian' akin to 'jacobian' or
// 'hessian') which is quite similar to a covariance matrix
template <typename scalar_t>
[[maybe_unused]] std::vector<scalar_t> grammian(const std::vector<scalar_t> &X,
                                                const size_t nrow,
                                                const bool intercept = false) {
  const size_t ncol = (X.size() / nrow) + intercept;
  std::vector<scalar_t> result(ncol * ncol, 0.0);
  size_t p = 0;
  if (intercept) {
    // if we are using intercept, the first row is easy, and we only need to do
    // ncol-1 additional operations, all of which amount to computing column
    // sums
    result[p++] = static_cast<scalar_t>(nrow);
    for (size_t i = 0; i < ncol - 1; i++) {
      scalar_t tmp = 0.0;
      for (size_t j = 0; j < nrow; j++) {
        tmp += X[i * nrow + j];
      }
      result[p++] = tmp;
    }
    // skip the first coefficient of second column; we already have this
    p++;
  }
  // the actual grammian ignoring intercepts
  for (size_t j = 0; j < ncol; j++) {
    for (size_t i = j; i < nrow; i++) {
      scalar_t tmp = 0.0;
      for (size_t k = 0; k < nrow; k++) {
        tmp += X[nrow * j + k] * X[nrow * i + k];
      }
      result[p++] = tmp;
    }
    // increment index - intercept is necessary here for correct offsetting
    p += j + 1 + intercept;
  }
  // symmetrize result
  for (size_t j = 1; j < ncol; j++) {
    for (size_t i = 0; i < j; i++) {
      result[j * ncol + i] = result[i * ncol + j];
    }
  }
  return result;
}
}  // namespace tinyqr::internal
namespace tinyqr {
template <typename scalar_t>
struct QR {
  std::vector<scalar_t> Q;
  std::vector<scalar_t> R;
};

template <typename scalar_t>
[[maybe_unused]] QR<scalar_t> qr_decomposition(const std::vector<scalar_t> &X,
                                               const size_t n, const size_t p,
                                               const scalar_t tol = 1e-8) {
  // initialize Q and R
  std::vector<scalar_t> Q(n * n, 0.0);
  // Q is an identity matrix
  for (size_t i = 0; i < n; i++) Q[i * n + i] = 1.0;
  std::vector<scalar_t> R = X;
  tinyqr::internal::qr_impl<scalar_t, true>(Q, R, n, p, tol);
  // keep in mind that only an n*p block of Q is meaningful, as well as only a
  // p*p block of R
  for (size_t i = 1; i < n; i++) {
    for (size_t j = 0; j < p; j++) {
      std::swap(Q[j * n + i], Q[i * n + j]);
    }
  }
  size_t k = 0;
  for (size_t i = 0; i < p; i++) {
    for (size_t j = 0; j < p; j++) {
      R[k++] = R[i * n + j];
    }
  }
  Q.resize(n * p);
  R.resize(p * p);
  return {Q, R};
}

template <typename scalar_t>
struct eigendecomposition {
  std::vector<scalar_t> eigenvals;
  std::vector<scalar_t> eigenvecs;
};

template <typename scalar_t>
[[maybe_unused]] eigendecomposition<scalar_t> qr_algorithm(
    const std::vector<scalar_t> &A, const size_t max_iter = 25,
    const scalar_t tol = 1e-8) {
  auto Ak = A;
  // A must be square
  const size_t n = std::sqrt(A.size());
  std::vector<scalar_t> QQ(n * n, 0.0);
  for (size_t i = 0; i < n; i++) QQ[i * n + i] = 1.0;
  // initialize Q and R
  std::vector<scalar_t> Q(n * n, 0.0);
  for (size_t i = 0; i < n; i++) Q[i * n + i] = 1.0;
  std::vector<scalar_t> R(n * n);  // = Ak;
  std::vector<scalar_t> temp(Q.size());
  for (size_t i = 0; i < max_iter; i++) {
    // reset Q and R, G gets reset inside qr_impl
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < n; k++) {
        // probably a decent way to reset to a diagonal matrix
        Q[j * n + k] = static_cast<scalar_t>(k == j);
        R[j * n + k] = Ak[j * n + k];
      }
    }
    // call QR decomposition
    tinyqr::internal::qr_impl<scalar_t, false>(Q, R, n, n, tol);
    tinyqr::internal::A_matmul_tB_to_C<scalar_t>(R, Q, Ak, n);
    // overwrite QQ in place
    size_t p = 0;
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < n; k++) {
        temp[p] = 0;
        for (size_t l = 0; l < n; l++) {
          temp[p] += QQ[l * n + k] * Q[l * n + j];
        }
        p++;
      }
    }
    // write to A colwise - i.e. directly
    for (size_t k = 0; k < QQ.size(); k++) {
      QQ[k] = temp[k];
    }
  }
  // diagonal elements of Ak are eigenvalues - we can just shuffle elements of A
  // and resize
  for (size_t i = 1; i < n; i++) {
    Ak[i] = Ak[i * n + i];
  }
  Ak.resize(n);
  return {Ak, QQ};
}
template <typename scalar_t>
class QRSolver {
  const size_t n;
  std::vector<scalar_t> Ak, QQ, Q, R, temp, eigval;

 public:
  explicit QRSolver<scalar_t>(const size_t n) : n(n) {
    this->Ak = std::vector<scalar_t>(n * n);
    this->QQ = std::vector<scalar_t>(n * n, 0.0);
    for (size_t i = 0; i < n; i++) this->QQ[i * n + i] = 1.0;
    // initialize Q and R
    this->Q = std::vector<scalar_t>(n * n, 0.0);
    for (size_t i = 0; i < n; i++) this->Q[i * n + i] = 1.0;
    this->R = std::vector<scalar_t>(n * n);
    this->temp = std::vector<scalar_t>(n * n);
    this->eigval = std::vector<scalar_t>(n);
  }
  void solve(const std::vector<scalar_t> &A, const size_t max_iter = 25,
             const scalar_t tol = 1e-8) {
    this->Ak = A;
    // in case we need to reset QQ
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
        this->QQ[i * n + j] = static_cast<scalar_t>(i == j);
      }
    }
    for (size_t i = 0; i < max_iter; i++) {
      // reset Q and R, G gets reset inside qr_impl
      for (size_t j = 0; j < n; j++) {
        for (size_t k = 0; k < n; k++) {
          // probably a decent way to reset to a diagonal matrix
          Q[j * n + k] = static_cast<scalar_t>(k == j);
          R[j * n + k] = Ak[j * n + k];
        }
      }
      // call QR decomposition
      tinyqr::internal::qr_impl<scalar_t>(Q, R, n, n, tol);
      tinyqr::internal::A_matmul_tB_to_C<scalar_t>(R, Q, Ak, n);
      // overwrite QQ in place
      size_t p = 0;
      for (size_t j = 0; j < n; j++) {
        for (size_t k = 0; k < n; k++) {
          temp[p] = 0;
          for (size_t l = 0; l < n; l++) {
            temp[p] += QQ[l * n + k] * Q[l * n + j];
          }
          p++;
        }
      }
      // write to A colwise - i.e. directly
      for (size_t k = 0; k < QQ.size(); k++) {
        QQ[k] = temp[k];
      }
    }
    for (size_t i = 0; i < n; i++) {
      eigval[i] = Ak[i * n + i];
    }
  }
  const std::vector<scalar_t> &eigenvalues() const { return eigval; }
  const std::vector<scalar_t> &eigenvectors() const { return this->QQ; }
};

// Function for back substitution
template <typename scalar_t>
[[maybe_unused]] std::vector<scalar_t> back_solve(
    const std::vector<scalar_t> &R, const std::vector<scalar_t> &y) {
  const size_t n = y.size();
  std::vector<scalar_t> result(n, 0.0);
  for (size_t i = n; i-- > 0;) {
    scalar_t temp = 0.0;
    for (size_t j = i + 1; j < n; ++j) {
      temp += R[j * n + i] * result[j];
    }
    result[i] = (y[i] - temp) / R[i * n + i];
  }
  return result;
}
// Function for back substitution using QR decomposition result
// this one is nicer since it saves us from computing an intermediary
template <typename scalar_t>
std::vector<scalar_t> back_solve(const std::vector<scalar_t> &Q,
                                 const std::vector<scalar_t> &R,
                                 const std::vector<scalar_t> &y,
                                 const size_t nrow, const size_t ncol) {
  std::vector<scalar_t> result(ncol, 0.0);
  for (size_t i = ncol; i-- > 0;) {
    scalar_t temp = 0.0;
    for (size_t j = i + 1; j < ncol; ++j) {
      temp += R[j * ncol + i] * result[j];
    }
    scalar_t y_tmp = 0;
    for (size_t j = 0; j < nrow; ++j) {
      // product Q'y need not be computed ahead of time; we can lazily compute
      // coefficient by coefficient, requiring only one temporary stack
      // allocated variable
      y_tmp += Q[i * nrow + j] * y[j];
    }
    result[i] = (y_tmp - temp) / R[i * ncol + i];
  }
  return result;
}
template <typename scalar_t>
[[maybe_unused]] std::vector<scalar_t> lm(const std::vector<scalar_t> &X,
                                          const std::vector<scalar_t> &y,
                                          const scalar_t tol = 1e-12) {
  const size_t nrow = y.size();
  const size_t ncol = X.size() / nrow;
  // compute QR decomposition
  const auto qr = tinyqr::qr_decomposition(X, nrow, ncol, tol);
  // solve Rx = Q'y
  return back_solve(qr.Q, qr.R, y, nrow, ncol);
}
}  // namespace tinyqr
#endif  // TINYQR_H_"
