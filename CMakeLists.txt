cmake_minimum_required(VERSION 3.27)
project(tinyqr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tinyqr main.cpp
        tinyqr.h)
