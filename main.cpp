#include "tinyqr.h"

#include <iostream>
int main() {
    // only works for symmetric matrices - which is my application (covariance matrices)
    const std::vector<double> A = {
            2.68, 8.86, 2.78, 0.09,
            8.86, 50.93, 3.78, 6.74,
            2.78, 3.78, 8.95, 2.94,
            0.09, 6.74, 2.94, 45.46};
    const auto res = tinyqr::qr_algorithm(A);
    std::cout<< "Eigenvals: \n";
    for(auto & eigval: res.eigenvals) {
        std::cout<< eigval << ",";
    }
    std::cout <<std::endl;
    std::cout<< "Eigenvecs: \n";
    for(size_t i = 0; i < 4; i++) {
        for(size_t j = 0; j < 4; j++) {
            std::cout <<res.eigenvecs[j*4 + i] << ',';
        }
        std::cout <<'\n';
    }
    std::cout << std::endl;
    return 0;
}