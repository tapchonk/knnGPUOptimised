cmake_minimum_required(VERSION 3.16)
project(KNNOriginal)
include(../../common.cmake)

add_executable(knnOriginal knnOriginal.cpp)
target_link_libraries(knnOriginal Kokkos::kokkos)