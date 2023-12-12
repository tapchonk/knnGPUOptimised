#include <cstdlib>
#include <random>
#include <utility>
#include <stdlib.h>
#include <Kokkos_Core.hpp>

#include "initialiseArrays.h"

#ifdef KOKKOS_ENABLE_CUDA

/**
 * @brief Initializes arrays used in K-nearest neighbors algorithm for CUDA execution.
 * 
 * @param nDimensionalSpace The view of the n-dimensional space array.
 * @param h_queryPointHost The view of the query point array on the host.
 * @param h_distancesHost The view of the distances array on the host.
 * @param h_classifierHost The view of the classifier array on the host.
 * @param N The number of data points.
 * @param D The dimensionality of the data points.
 * @param C The number of times to repeat the classification process.
 * @param NC The number of classes.
 * @param partitions The number of partitions for the n-dimensional space array.
 * @param partitionD The dimensionality of each partition.
 * 
 * @return void
 */
void initialiseArrays(  NDimensionalSpaceView nDimensionalSpace, HostQueryPointView h_queryPointHost, HostDistanceView h_distancesHost, ClassifierView h_classifierHost,
                        long long N, long long D, long long C, long long NC, long long partitions, long long partitionD ) {

  srand(42); //Seed the random number generator for consistent results

  for (int repeats = 0; repeats < C; repeats++) {
    //Generate a random set of points to query
    for (int i = 0; i < D; i++) {
      h_queryPointHost(repeats, i) = ((float) rand() / (RAND_MAX)) * 2000.0f - 1000.0f;
    }
  }

  //Fill arrays with random floats that range between -1000.0 and 1000.0
  //We need to do this for each partition of the nDimensionalSpace array
  for (long long partition = 0; partition < partitions; partition++) {
    long long part = partitionD;
    if ((partition == partitions - 1) && (partitions > 1)) {
      part = D % partitionD;
    }
    if (D < partitionD) {
      part = D;
    }
    for (long long i = 0; i < part; i++) {
      for (long long j = 0; j < N; j++) {
        nDimensionalSpace(partition, i, j) = ((float) rand() / (RAND_MAX)) * 2000.0f - 1000.0f;
      }
    }
  }

  //Fill array with random integers between 0 and NC to represent our "classes"

  for (int i = 0; i < N; i++) {
    h_classifierHost( 0, i ) = ((int) rand() % NC);
  }

  //Repeat this copy for each set of data we want to classify

  for (int repeats = 1; repeats < C; repeats++) {
    for (int i = 0; i < N; i++) {
      h_classifierHost(repeats, i) = h_classifierHost(0, i);
    }
  }

  //Set all distances to be sorted to 0.0f

  for (int repeats = 0; repeats < C; repeats++) {
    for (int i = 0; i < N; i++) {
      h_distancesHost(repeats, i) = 0.0f;
    }
  }
}

#endif

#ifdef KOKKOS_ENABLE_OPENMP

/**
 * @brief Initializes arrays used in K-nearest neighbors algorithm for OpenMP execution.
 * 
 * @param nDimensionalSpace The view of the n-dimensional space array on the host.
 * @param queryPoint The view of the query point array on the host.
 * @param distances The view of the distances array on the host.
 * @param classifier The view of the classifier array.
 * @param N The number of data points.
 * @param D The dimensionality of the data points.
 * @param C The number of times to repeat the classification process.
 * @param NC The number of classes.
 * 
 * @return void
 */
void initialiseArrays(  NDimensionalPartitionView nDimensionalSpace, QueryPointView queryPoint, DistanceView distances, ClassifierView classifier,
                        long long N, long long D, long long C, long long NC ) {

  srand(42); //Seed the random number generator for consistent results

  for (int repeats = 0; repeats < C; ++repeats) {
    //Generate a random set of points to query
    for (int i = 0; i < D; ++i) {
      queryPoint(repeats, i) = ((float) rand() / (RAND_MAX)) * 2000.0f - 1000.0f;
    }
  }

  //Fill arrays with random floats that range between -1000.0 and 1000.0

  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < N; ++j) {
      nDimensionalSpace(i, j) = ((float) rand() / (RAND_MAX)) * 2000.0f - 1000.0f;
    }
  }

  //Fill array with random integers between 0 and NC to represent our "classes"

  for (int i = 0; i < N; ++i) {
    classifier( 0, i ) = ((int) rand() % NC);
  }

  //Repeat this copy for each set of data we want to classify

  for (int repeats = 1; repeats < C; ++repeats) {
    for (int i = 0; i < N; ++i) {
      classifier(repeats, i) = classifier(0, i);
    }
  }

  //Set all distances to be sorted to 0.0f

  for (int repeats = 0; repeats < C; ++repeats) {
    for (int i = 0; i < N; i++) {
      distances(repeats, i) = 0.0f;
    }
  }
}

#endif
