cmake_minimum_required(VERSION 3.12)
project(gillespie)

find_package(pybind11 REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

add_library(gillespie MODULE simulate_gillespie_cpp.cpp)
include_directories("/usr/include/eigen3/")
target_link_libraries(gillespie PRIVATE pybind11::pybind11)