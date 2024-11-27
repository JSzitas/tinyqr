### Tiny QR decomposition and linear solver

This library is what it says on the tin.
Simply include the header and run:

```c++
#include "../tinyqr.h"

int main() {
  const std::vector<double> A = {2.68, 8.86, 2.78, 0.09,
                                 8.86, 50.93, 3.78, 6.74,
                                 2.78, 3.78, 8.95, 2.94,
                                 0.09, 6.74, 2.94, 45.46};
  const auto QR_result = tinyqr::qr_decomposition(X, 4, 4);
```

Note that this library also has the underlying QR algorithm (duh) and can solve linear systems using QR: 

```c++
#include "../tinyqr.h"

int main() {
    // note that A is in column major format; so you are seeing the columns laid out
    const std::vector<double> A = {2.68, 8.86, 2.78, 0.09, 8.86, 50.93, 3.78, 6.74,
                                   2.78, 3.78, 8.95, 2.94, 0.09, 6.74, 2.94, 45.46};
    const std::vector<double> y = {-1.838, -7.380, 0.9752,1.948, -10.955, -54.305, -3.484, 13.861};
    // note that A is in column major format; so you are seeing the columns laid out
    // sizes are inferred automatically
    const auto coefficients = tinyqr::lm(X, y);
```

Funnily enough, both can also be done at compile time; you just need constexpr arrays :) 

```c++
#include "../tinyqr.h"

int main() {
    constexpr std::array<double, 4*4> A = {2.68, 8.86, 2.78, 0.09,
                                      8.86, 50.93, 3.78, 6.74,
                                      2.78, 3.78, 8.95, 2.94,
                                      0.09, 6.74, 2.94, 45.46
    }};
    // note cqr, not qr!
    const auto QR_result = tinyqr::cqr_decomposition(X, 4, 4);
```

Note that this library also has the underlying QR algorithm (duh) and can solve linear systems using QR:

```c++
#include "../tinyqr.h"

int main() {
    // 8*2 just for the benefit of the reader
    constexpr std::array<double, 8*2> A = {2.68, 8.86, 2.78, 0.09, 8.86, 50.93, 3.78, 6.74,
                                           2.78, 3.78, 8.95, 2.94, 0.09, 6.74, 2.94, 45.46};
    constexpr std::array<double, 8> y = {-1.838, -7.380, 0.9752,1.948, -10.955, -54.305, -3.484, 13.861};
    // note clm, not lm
    const auto coefficients = tinyqr::clm(X, y);
```

This library uses Givens rotations as the underlying method, and has some decent SIMD based optimizations. 
Expect it to be modestly fast, and usually not cripplingly slow.

The compile time stuff is wholly unoptimized; I needed it for relatively simple cases where I did 
not want to just implement caching. Your mileage might vary, but it should be fairly easy to modify if you need 
a fast compile time solver. 


