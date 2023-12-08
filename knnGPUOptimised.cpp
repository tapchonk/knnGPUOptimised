/**
 * @file knnGPUOptimised.cpp
 * @brief Heterogeneous compute version of the KNN algorithm.
 *
 * This program implements the K-nearest neighbors (KNN) algorithm using heterogeneous compute resources.
 * It takes command line arguments to specify the number of data points, dimensions, nearest neighbors,
 * sets of data to classify, and number of classes. The program performs initialization, calculates
 * n-dimensional Euclidean distances, and sorts the distances and classifier arrays. The final result
 * is stored in the finalClasses array.
 * 
 * For example to run the application with 1000 data points, 2 dimensions, 10 nearest neighbors, 10 sets of data to classify, and 10 classes:
 * ./knnGPUOptimised -N 1000 -D 2 -K 10 -C 10 -NC 10
 * You should really run this on a batch compute system to get accurate results for timings.
 * As such I would recommend the remote run script provided and editing it for your specific batch compute system.
 * and then run the script with:
 * ./remoterun.sh ./knnGPUOptimised -N 1000 -D 2 -K 10 -C 10 -NC 10
 */

//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology  Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
// We are making a copy of the original file, so we can compare the three.
// This will be the Heterogeneous compute version of the KNN algorithm.

#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <utility>
#include <stdlib.h>

#include "checkSizes.h"
#include "initialiseArrays.h"
#include "nDimensionalEuclid.h"
#include "parallelisedMergeSort.h"

#ifdef USING_THRUST
  #include <thrust/sort.h>
  #include <thrust/execution_policy.h>
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_NestedSort.hpp>


/**
 * @brief This program takes command line arguments to specify the number of data points, dimensions, nearest neighbors,
 * sets of data to classify, and number of classes. It reads the command line arguments and assigns them to the corresponding variables.
 * Then, it generates random query points based on the specified number of data points and dimensions.
 * The program performs initialisation, calculates n-dimensional Euclidean distances between the query points and the data points,
 * and sorts the distances and classifier arrays. Finally, it outputs the classes for each of the randomly generated query points.
 *
 * @param N The number of data points.
 * @param D The dimensionality of the data points.
 * @param K The number of nearest neighbors used to classify our querypoints.
 * @param C The number of times to repeat the classification process.
 * @param NC The number of classes.
 *  
 * @return int 
 */
int main( int argc, char* argv[] )
{
  long long N = -1;         // number of data points
  long long D = -1;         // number of dimensions
  long long K = -1;         // number of nearest neighbours
  long long C = -1;         // number of sets of data to classify
  long long NC = -1;        // number of classes

  std::chrono::duration<double> initialisationTime;
  std::chrono::duration<double> nDimensionalEuclidTime;
  std::chrono::duration<double> classifierMergeSortTime;

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Data Points" ) == 0 ) ) {
      N = atoi( argv[ ++i ] );
      printf( "  User N is %d\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-D" ) == 0 ) || ( strcmp( argv[ i ], "-Dimensions" ) == 0 ) ) {
      D = atoi( argv[ ++i ] );
      printf( "  User D is %d\n", D );
    }
    else if ( ( strcmp( argv[ i ], "-K" ) == 0 ) || ( strcmp( argv[ i ], "-Neighbours" ) == 0 ) ) {
      K = atoi( argv[ ++i ] ) ;
      printf( "  User K is %d\n", K );
    }
    else if ( ( strcmp( argv[ i ], "-C" ) == 0 ) || ( strcmp( argv[ i ], "-Classifications" ) == 0 ) ) {
      C = atoi( argv[ ++i ] ) ;
      printf( "  User C is %d\n", C );
    }
    else if ( ( strcmp( argv[ i ], "-NC" ) == 0 ) || ( strcmp( argv[ i ], "-Number of Classes" ) == 0 ) ) {
      NC = atoi( argv[ ++i ] ) ;
      printf( "  User NC is %d\n", NC );
    }

    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      fprintf(stdout,  "  KNN Options:\n" );
      fprintf(stdout,  "  -Data Points       (-N ) <int>:   num, determines number of data points (num) (default: 1000)\n" );
      fprintf(stdout,  "  -Dimensions        (-D ) <int>:   num, determines number of arrays to allocate (num) (default: 2)\n" );
      fprintf(stdout,  "  -K-Neighbours      (-K ) <int>:   num, determines number of neighbours to use as the classifier sample (num) (default: 10)\n" );
      fprintf(stdout,  "  -Classes           (-C ) <int>:   num, determines number of classes to use as the classifier sample (num) (default: 10)\n" );
      fprintf(stdout,  "  -Number of Classes (-NC) <int>:   num, determines number of classes to use as the classifier sample (num) (default: 10)\n" );
      fprintf(stdout,  "  -help              (-h ):         print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( N, D, K, C, NC);

  Kokkos::initialize( argc, argv );
  {

  int numThreads;
  if (getenv("OMP_NUM_THREADS"))
    numThreads = std::atoi(getenv("OMP_NUM_THREADS"));
  fprintf(stdout, "\n================================================================================\n");
  fprintf(stdout, "Number of Threads is : %d.\n", numThreads);

  #ifdef KOKKOS_ENABLE_OPENMP
    long long partitionD = D;
    long long partitions = 1;
  #endif

  #ifdef KOKKOS_ENABLE_CUDA
    size_t totalGPUMem = 0;
    size_t totalFreeGPUMem = 0;
    cudaMemGetInfo ( &totalFreeGPUMem, &totalGPUMem );
    fprintf(stdout, "\n================================================================================\n");
    fprintf(stdout, "Free GPU Memory is : %zu (bytes).\n", totalFreeGPUMem);
    
    long long spaceTotal = ((D * N) + (C * D) + (3 * C * N) + (2 * N) + C + NC) * sizeof(float);

    fprintf(stdout, "Total Memory usage is : %ld (bytes).\n", spaceTotal);

    long long gpuTotal = ((C * N) + (C * D) + (3 * N)) * sizeof(float);
    // the 3 * N is for padding for thrust sort
    fprintf(stdout, "Total GPU Memory usage is : %ld (bytes before N-Dimensional allocation).\n", gpuTotal);

    fprintf(stdout, "\n================================================================================\n");

    long long partitionD = ((static_cast<long long>(totalFreeGPUMem) - gpuTotal)/sizeof(float))/N;

    long long partitions = D/partitionD + 1;
    if (totalFreeGPUMem < spaceTotal) {
      //spaceTotal = (N/partitions)*D + spaceTotal;
      fprintf(stdout, "WARNING: Application memory usage is larger than total GPU memory available.\nPartitioning workload amongst %d partitions.\n", partitions);
    } else {
      partitionD = D;
    }
  #endif

  #ifdef KOKKOS_ENABLE_CUDA
    #define MemSpace Kokkos::CudaSpace
    #define MemLayout Kokkos::LayoutRight
  #endif
  #ifdef KOKKOS_ENABLE_OPENMP
    #define MemSpace Kokkos::OpenMP
    #define MemLayout Kokkos::LayoutLeft
    using openmp_range_policy = Kokkos::RangePolicy<Kokkos::OpenMP::execution_space>;
  #endif
  #ifdef KOKKOS_ENABLE_HIP // (if we want to add support for Radeon GPUs later)
    #define MemSpace Kokkos::Experimental::HIPSpace
    #define MemLayout Kokkos::LayoutRight
  #endif
  #ifndef MemSpace
    #define MemSpace Kokkos::HostSpace
    #define MemLayout Kokkos::LayoutLeft
  #endif

  using ExecutionSpace = MemSpace::execution_space;
  //using current_range_policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Static>>;
  //We can use this "using TeamPol = Kokkos::TeamPolicy<ExecutionSpace>;" to define team policy ranges
  //                "using TeamMem = typename TeamPol::member_type;" defines the member device of the execution policy

  //Start timers for the algorithm
  auto start = std::chrono::high_resolution_clock::now();

  //Allocate memory space for the arrays
  Kokkos::View<float**, MemLayout, MemSpace> nDimensionalSpacePartition( "nDimensionalSpacePartition", partitionD, N );
  Kokkos::View<float**, Kokkos::LayoutRight, MemSpace> queryPoint( "queryPoint", C, D );
  Kokkos::View<float**, Kokkos::LayoutRight, MemSpace> distances( "distances", C, N );
  Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace> classifier( "classifier", C, N );
  //We can use ", Kokkos::MemoryTraits<Kokkos::RandomAccess>" to take advantage of the stacked HBM2 memory which optimises for random access due to non-contiguous memory access

  //define host mirrors between the Host and GPU device
  Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_nDimensionalSpacePartitionHost = Kokkos::create_mirror_view( nDimensionalSpacePartition );
  Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>::HostMirror h_queryPointHost = Kokkos::create_mirror_view( queryPoint );
  Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>::HostMirror h_distancesHost = Kokkos::create_mirror_view( distances );

  //to store the classes results
  Kokkos::View<int*, Kokkos::HostSpace> finalClasses( "finalClasses", C );
  Kokkos::View<int*, Kokkos::HostSpace> classifierCount( "classifierCount", NC );

  #ifdef KOKKOS_ENABLE_OPENMP
    //just for a placeholder in this case
    Kokkos::View<float***, Kokkos::LayoutRight, MemSpace> nDimensionalSpace( "nDimensionalSpace", 1, 1, 1 );
    //allocate arrays for merge sort rather than allocating memory from the stack
    Kokkos::View<float*, MemSpace> temporaryOne( "temporaryOne", N );
    Kokkos::View<int*, MemSpace> temporaryTwo( "temporaryTwo", N );
    initialiseArrays(nDimensionalSpacePartition, queryPoint, distances, classifier, N, D, C, NC );
  #endif

  #ifdef KOKKOS_ENABLE_CUDA
    //to store the N-Dimensional space partitions
    Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace> nDimensionalSpace( "nDimensionalSpace", partitions,  partitionD, N );
    initialiseArrays(nDimensionalSpace, h_queryPointHost, h_distancesHost, classifier, N, D, C, NC, partitions, partitionD);
    Kokkos::deep_copy(distances, h_distancesHost);
    Kokkos::deep_copy(queryPoint, h_queryPointHost);
    //immediately copy the first partition of the nDimensionalSpace array to the GPU
    //or in the case where the entirety of the data can fit on the GPU, copy the entire space
    auto currWorkload = Kokkos::subview(nDimensionalSpace, 0, Kokkos::ALL(), Kokkos::ALL());
    Kokkos::deep_copy(nDimensionalSpacePartition, currWorkload);
    //Kokkos::deep_copy automatically fences so that the host and device are synchronised
    Kokkos::fence();
  #endif
  //End timer for array allocation
  auto end = std::chrono::high_resolution_clock::now();

  //Store the time it took to allocate the arrays
  initialisationTime = end - start;

  //Start timer for the nDimensionalEuclid function
  start = std::chrono::high_resolution_clock::now();

  nDimensionalEuclid( nDimensionalSpace, queryPoint, distances, nDimensionalSpacePartition, N, D, C, partitions, partitionD );

  //End timer for the nDimensionalEuclid function
  end = std::chrono::high_resolution_clock::now();

  //store the time it took to run the nDimensionalEuclid function
  nDimensionalEuclidTime += end - start;

  start = std::chrono::high_resolution_clock::now();

  //we now need the classifier array on the GPU so now we create a new view for it and copy over the data
  #ifdef KOKKOS_ENABLE_CUDA
    nDimensionalSpacePartition = distances;
    Kokkos::View<int**, Kokkos::LayoutRight, MemSpace> classifierGPU( "classifierGPU", C, N );
    Kokkos::deep_copy(classifierGPU, classifier);
  #endif

  end = std::chrono::high_resolution_clock::now();

  initialisationTime += end - start;
    
  //Start timer for the classifier Sort function
  start = std::chrono::high_resolution_clock::now();

  #ifdef KOKKOS_ENABLE_CUDA
  //sort the distances array and the classifier array by the distances array, repeats this for the number of query points to classify
    for (int repeats = 0; repeats < C; ++repeats) {
      //get the subviews of the distances and classifier arrays for the current query point
      auto distanceSub = Kokkos::subview(distances,  repeats, Kokkos::ALL());
      auto classifierSub = Kokkos::subview(classifierGPU, repeats, Kokkos::ALL());
      //utilise thrust sort by key to sort the classifier array using the distance as a key
      thrust::sort_by_key(thrust::device, distanceSub.data(), distanceSub.data() + N, classifierSub.data());
    }
    Kokkos::deep_copy(classifier, classifierGPU);
  #endif 

  #ifdef KOKKOS_ENABLE_OPENMP
    //utilise the parallelised merge sort algorithm to sort the classifier array by the distances array
    for (int repeats = 0; repeats < C; ++repeats) {
      parallelisedMergeSort(distances, classifier, N, temporaryOne, temporaryTwo, repeats, numThreads);
    }
  #endif
  
  //simple loop to identify the K closest neighbours and count the number of each class within the K-Nearest set
  for (int repeats = 0; repeats < C; ++repeats) {
    for (int i = 0; i < NC; ++i) {
      classifierCount(i) = 0;
    }

    for (int i = 0; i < K; ++i) {
      classifierCount(classifier( repeats, i )) += 1;  
    }

    int max = 0;
    int maxIndex = 0;
    for (int i = 0; i < NC; ++i) {
      if (classifierCount(i) > max) {
        max = classifierCount(i);
        maxIndex = i;
      }
    }
    finalClasses(repeats) = maxIndex;
  }
  
  //End timer for the classifierMergeSort function
  end = std::chrono::high_resolution_clock::now();

  //Store the time it took to run the classifierMergeSort function
  classifierMergeSortTime += end - start;
  
  auto totalTime = initialisationTime + nDimensionalEuclidTime + classifierMergeSortTime;

  //output results
  fprintf(stdout, "\n||============================RESULTS============================||\n");
  fprintf(stdout, "||Initialisation Time:                                %.6f   ||\n", initialisationTime.count());
  fprintf(stdout, "||N-Dimensional Euclid Time:                          %.6f   ||\n", nDimensionalEuclidTime.count());
  fprintf(stdout, "||Sort Time:                                          %.6f   ||\n", classifierMergeSortTime.count());
  fprintf(stdout, "||Total Time:                                         %.6f   ||\n", totalTime.count());
  fprintf(stdout, "||============================RESULTS============================||\n");
  for (int i = 0; i < C; i++) {
    fprintf(stdout, "||Data Point Set %d:                                     %d        ||\n", i, finalClasses(i));
  }
  fprintf(stdout, "||============================RESULTS============================||\n");

  }
  Kokkos::finalize();
  
  return 0;
}