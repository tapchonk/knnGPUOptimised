#ifndef NDIMENSIONALEUCLID_H
#define NDIMENSIONALEUCLID_H
#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA

#define NDimensionalSpaceView Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace>
#define NDimensionalPartitionView Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>
#define QueryPointView Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>
#define DistanceView Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>

#endif

#ifdef KOKKOS_ENABLE_OPENMP

#define NDimensionalSpaceView Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace>
#define NDimensionalPartitionView Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::OpenMP>
#define QueryPointView Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::OpenMP>
#define DistanceView Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::OpenMP>

#endif

void nDimensionalEuclid(  NDimensionalSpaceView nDimensionalSpace, QueryPointView queryPoint, DistanceView distances, NDimensionalPartitionView nDimensionalPartition,
                          long long N, long long D, long long C, long long partitions, long long partitionD );


#endif