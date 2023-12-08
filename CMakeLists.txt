cmake_minimum_required(VERSION 3.16)
project(KNNGPUOptimised)
include(../../common.cmake)

add_executable(knnGPUOptimised knnGPUOptimised.cpp)
target_link_libraries(knnGPUOptimised Kokkos::kokkos)