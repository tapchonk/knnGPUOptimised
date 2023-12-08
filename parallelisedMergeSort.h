#ifndef PARALLELISEDMERGESORT_H
#define PARALLELISEDMERGESORT_H
#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_OPENMP
#define DistanceView Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::OpenMP>
#define ClassifierView Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::OpenMP>
#define TemporaryOneView Kokkos::View<float*, Kokkos::OpenMP>
#define TemporaryTwoView Kokkos::View<int*, Kokkos::OpenMP>

void parallelisedMergeSort(  DistanceView distances, ClassifierView classifier, long long N, TemporaryOneView temporaryOne, TemporaryTwoView temporaryTwo, int repeats, int numThreads );
#endif
#endif
