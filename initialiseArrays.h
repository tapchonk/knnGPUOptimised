#ifndef INITIALISEARRAYS_H
#define INITIALISEARRAYS_H
#include <Kokkos_Core.hpp>


#ifdef KOKKOS_ENABLE_CUDA

#define NDimensionalSpaceView Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace>
#define HostQueryPointView Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>::HostMirror
#define HostDistanceView Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>::HostMirror
#define ClassifierView Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>
void initialiseArrays(  NDimensionalSpaceView nDimensionalSpace, HostQueryPointView h_queryPointHost, HostDistanceView h_distancesHost, ClassifierView h_classifierHost,
                        long long N, long long D, long long C, long long NC, long long partitions, long long partitionD );

#endif

#ifdef KOKKOS_ENABLE_OPENMP

#define NDimensionalPartitionView Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::OpenMP>
#define QueryPointView Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::OpenMP>
#define DistanceView Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::OpenMP>
#define ClassifierView Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::OpenMP>
void initialiseArrays(  NDimensionalPartitionView nDimensionalSpace, QueryPointView queryPoint, DistanceView distances, ClassifierView classifier,
                        long long N, long long D, long long C, long long NC );

#endif

#endif
