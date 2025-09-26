cmake_minimum_required(VERSION 3.27)
project(tinyqr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(benchmark REQUIRED)
message(STATUS \"benchmark version: ${benchmark_VERSION}\")
add_executable(benchmark_lm benchmarking/benchmark_lm.cpp)
target_compile_options(benchmark_lm PRIVATE -O3 -march=native)
target_link_libraries(benchmark_lm PRIVATE tinyqr
         PRIVATE benchmark::benchmark)

add_custom_target(multi_benchmark_lm
        COMMAND $<TARGET_FILE:benchmark_lm>
        --benchmark_repetitions=10
        DEPENDS benchmark_lm
        COMMENT "Running benchmarks with multiple repetitions"
)

add_executable(profile_lm benchmarking/profile_lm.cpp)
target_compile_options(profile_lm PRIVATE -O2 -pg -ggdb -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -Wall)

add_library(tinyqr main.cpp tinyqr.h)

# add_executable(tinyqr_prof benchmarking/benchmark_lm.cpp)
# target_compile_options(tinyqr_prof PRIVATE -pg -O2 -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer)
# target_link_libraries(tinyqr_prof PRIVATE tinyqr)