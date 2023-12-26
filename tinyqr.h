#include <vector>
#include <limits>
#include <cmath>
#include <array>
#include <algorithm>

namespace {
    template<typename scalar_t>
    std::tuple<scalar_t, scalar_t>
    givens_rotation(scalar_t a, scalar_t b) {
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

    template<typename scalar_t, const size_t max_size = 100>
    void A_matmul_B_to_A(
            std::vector<scalar_t> &A,
            std::vector<scalar_t> &B,
            const size_t ncolA,
            const size_t ncolB) {
        std::array<scalar_t, max_size> temp; // NOLINT
        const size_t nrowA = A.size() / ncolA;
        const size_t nrowB = B.size() / ncolB;
        size_t p = 0;
        for (size_t j = 0; j < ncolB; j++) {
            for (size_t i = 0; i < ncolA; i++) {
                temp[p] = 0;
                for (size_t k = 0; k < ncolA; k++) {
                    temp[p] += A[k * nrowA + i] * B[j * nrowB + k];
                }
                p++;
            }
        }
        // write to A colwise - i.e. directly
        for (size_t i = 0; i < A.size(); i++) {
            A[i] = temp[i];
        }
    }

    template<typename scalar_t, const size_t max_size = 100>
    [[maybe_unused]] void A_matmul_B_to_B(std::vector<scalar_t> &A, std::vector<scalar_t> &B,
                                          const size_t ncol) {
        std::array<scalar_t, max_size> temp;
        // carry out multiplications
        for (size_t k = 0; k < ncol; k++) {
            for (size_t l = 0; l < ncol; l++) {
                scalar_t accumulator = 0.0;
                for (size_t m = 0; m < ncol; m++) {
                    accumulator += A[m * ncol + k] * B[l * ncol + m];
                }
                temp[l * ncol + k] = accumulator;
            }
        }
        for (size_t k = 0; k < B.size(); k++) {
            B[k] = temp[k];
        }
    }

    template<typename scalar_t, const size_t max_size = 100>
    void A_matmul_B_to_C(
            std::vector<scalar_t> &A,
            std::vector<scalar_t> &B,
            std::vector<scalar_t> &C,
            const size_t ncol) {
        std::array<scalar_t, max_size> temp; // NOLINT
        // carry out multiplications
        for (size_t k = 0; k < ncol; k++) {
            for (size_t l = 0; l < ncol; l++) {
                scalar_t accumulator = 0.0;
                for (size_t m = 0; m < ncol; m++) {
                    accumulator += A[m * ncol + k] * B[l * ncol + m];
                }
                temp[l * ncol + k] = accumulator;
            }
        }
        for (size_t k = 0; k < C.size(); k++) {
            C[k] = temp[k];
        }
    }

    template<typename scalar_t, const size_t max_size = 100>
    void tA_matmul_B_to_B(std::vector<scalar_t> &A, std::vector<scalar_t> &B,
                          const size_t ncolA, const size_t ncolB) {
        std::array<scalar_t, max_size> temp;  //NOLINT
        const size_t nrowA = A.size() / ncolA;
        size_t p = 0;
        for (size_t j = 0; j < ncolB; j++) {
            for (size_t i = 0; i < ncolA; i++) {
                temp[p] = 0.0;
                for (size_t k = 0; k < ncolA; k++) {
                    temp[p] += A[(i * nrowA) + k] * B[(j * ncolA) + k];
                }
                p++;
            }
        }
        p = 0;
        for (size_t i = 0; i < ncolA; i++) {
            for (size_t j = 0; j < ncolB; j++) {
                B[i * ncolA + j] = temp[p++];
            }
        }
    }

    // this is the implementation of QR decomposition - this does not get exposed, only the
    // nice(r) facades do
    template<typename scalar_t, const size_t max_size = 100>
    void
    qr_impl(std::vector<scalar_t> &Q,
            std::vector<scalar_t> &R,
            std::vector<scalar_t> &G,
            const size_t n,
            const scalar_t tol) {
        for (size_t j = 0; j < n; j++) {
            for (size_t i = n - 1; i > j; i--) {
                // using tuples and structured bindings should make this fairly ok performance wise
                const auto [c, s] = givens_rotation(R[(j * n) + (i - 1)], R[j * n + i]);
                // this replaces a block of size 2x2 using the rotation
                G[((i - 1) * n) + (i - 1)] = c;
                G[((i - 1) * n) + i] = s;
                G[(i * n) + (i - 1)] = -s;
                G[(i * n) + i] = c;
                tA_matmul_B_to_B<scalar_t, max_size>(G, R, n, n);
                A_matmul_B_to_A<scalar_t, max_size>(Q, G, n, n);
                // overwrite the 2x2 block of previously rotated values
                G[((i - 1) * n) + (i - 1)] = 1.0;
                G[((i - 1) * n) + i] = 0.0;
                G[(i * n) + (i - 1)] = 0.0;
                G[(i * n) + i] = 1.0;
            }
        }
        // clean up R - particularly under the diagonal
        for (auto &val: R) {
            val = std::abs(val) < tol ? 0.0 : val;
        }
    }
} // namespace
namespace tinyqr {
    template<typename scalar_t>
    struct QR {
        std::vector<scalar_t> Q;
        std::vector<scalar_t> R;
    };

    template<typename scalar_t, const size_t max_size = 100>
    [[maybe_unused]] QR<scalar_t> qr_decomposition(const std::vector<scalar_t> &A,
                                                   const scalar_t tol = 1e-8) {
        const size_t n = std::sqrt(A.size());
        // initialize Q and R
        std::vector<scalar_t> Q(n * n, 0.0);
        // Q is an identity matrix
        for (size_t i = 0; i < n; i++) {
            Q[i * n + i] = 1.0;
        }
        std::vector<scalar_t> R = A;
        // initialize G
        std::vector<scalar_t> G(n * n, 0.0);
        for (size_t i = 0; i < n; i++) {
            G[i * n + i] = 1.0;
        }
        qr_impl<scalar_t, max_size>(Q, R, G, n, tol);
        return {Q, R};
    }

    template<typename scalar_t>
    struct eigendecomposition {
        std::vector<scalar_t> eigenvals;
        std::vector<scalar_t> eigenvecs;
    };

    template<typename scalar_t, const size_t max_size = 100>
    [[maybe_unused]] eigendecomposition<scalar_t> qr_algorithm(
            const std::vector<scalar_t> &A,
            const size_t max_iter = 35,
            const scalar_t tol = 1e-8) {
        auto Ak = A;
        // A is square
        const size_t n = std::sqrt(A.size());
        std::vector<scalar_t> QQ(n * n, 0.0);
        for (size_t i = 0; i < n; i++) {
            QQ[i * n + i] = 1.0;
        }
        // initialize Q and R
        std::vector<scalar_t> Q(n * n, 0.0);
        for (size_t i = 0; i < n; i++) {
            Q[i * n + i] = 1.0;
        }
        std::vector<scalar_t> R = Ak;
        std::vector<scalar_t> G(n * n, 0.0);
        for (size_t i = 0; i < n; i++) {
            G[i * n + i] = 1.0;
        }
        for (size_t i = 0; i < max_iter; i++) {
            // reset Q and R, G gets reset inside qr_impl
            for (size_t j = 0; j < n; j++) {
                for (size_t k = 0; k < n; k++) {
                    // probably a decent way to reset to a diagonal matrix
                    Q[j * n + k] = static_cast<scalar_t>(k == j);
                    // and this is really just a direct copy, but since we have this loop anyway
                    R[j * n + k] = Ak[j * n + k];
                }
            }
            // call QR decomposition
            qr_impl<scalar_t, max_size>(Q, R, G, n, tol);
            A_matmul_B_to_C<scalar_t, max_size>(R, Q, Ak, n);
            A_matmul_B_to_A<scalar_t, max_size>(QQ, Q, n, n);
        }
        // diagonal elements of Ak are eigenvalues
        std::vector<scalar_t> eigvals(n);
        for (size_t i = 0; i < n; i++) {
            eigvals[i] = Ak[i * n + i];
        }
        return {eigvals, QQ};
    }
}
