//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
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
// We are making a copy of the original file, so we can compare the two.
// This will the standard version of the KNN algorithm.

#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <stdlib.h>

#include <Kokkos_Core.hpp>

void checkSizes( long long &N, long long &D, long long &K, long long &C, long long &NC);

void mergeSort( float * distances, int * classifier, int lPointer, int rPointer);

void mergeSort( float * distances, int * classifier, int lPointer, int rPointer, float * temporaryOne, int * temporaryTwo);

void merge( float * distances, int * classifier, int lPointer, int mPointer, int rPointer, float * temporaryOne, int * temporaryTwo);

void initialiseArrays( float * nDimensionalSpace, int * classifier, float * queryPoint, long long &D, long long &N, long long &C, long long &NC);

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
      fprintf(stdout,  "  -Data Points (-N) <int>:   num, determines number of data points (num) (default: 1000)\n" );
      fprintf(stdout,  "  -Dimensions (-D) <int>:    num, determines number of arrays to allocate (num) (default: 2)\n" );
      fprintf(stdout,  "  -K-Neighbours (-K) <int>:    num, determines number of neighbours to use as the classifier sample (num) (default: 10)\n" );
      fprintf(stdout,  "  -Classes (-C) <int>:       num, determines number of classes to use as the classifier sample (num) (default: 10)\n" );
      fprintf(stdout,  "  -Number of Classes (-NC) <int>:       num, determines number of classes to use to sort each data point to (num) (default: 10)\n" );
      fprintf(stdout,  "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( N, D, K, C, NC);

  //Start timers for the algorithm
  auto start = std::chrono::high_resolution_clock::now();
  //Allocate memory for the arrays
  auto nDimensionalSpace = static_cast<float*>(std::malloc(N * D * sizeof(float)));
  auto queryPoint = static_cast<float*>(std::malloc(D * C * sizeof(float)));
  auto distances = static_cast<float*>(std::malloc(N * sizeof(float)));
  //The classifier is arbitrary, it's not a focal point of this project
  auto currentClassifier = static_cast<int*>(std::malloc(N * sizeof(int)));
  auto classifier = static_cast<int*>(std::malloc(N * sizeof(int)));
  auto temporaryOne = static_cast<float*>(std::malloc((N) * sizeof(float)));
  auto temporaryTwo = static_cast<int*>(std::malloc((N) * sizeof(int)));
  auto finalClasses = static_cast<int*>(std::malloc((C) * sizeof(int)));
  auto classifierCount = static_cast<int*>(std::malloc(NC * sizeof(int)));
  
  // Kokkos::initialize( argc, argv );
  
  initialiseArrays(nDimensionalSpace, classifier, queryPoint, D, N, C, NC);

  //End timer for array allocation
  auto end = std::chrono::high_resolution_clock::now();

  //Store the time it took to allocate the arrays
  initialisationTime = end - start;

  for (int repeats = 0; repeats < C; repeats++) {

    //Start timer for the nDimensionalEuclid function
    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        distances[i] = 0.0;
    }

    for (int i = 0; i < N; i++) {
        currentClassifier[i] = classifier[i];
    }

    //Calculate the euclidean distance between the query point and each point in the nDimensionalSpace
    //Loop interchange was a good idea here, as it allows for improved memory locality
    float diff;
    for (int i = 0; i < D; i++) {
      for (int j = 0; j < N; j++) {
        diff = queryPoint[repeats * D + i] - nDimensionalSpace[i * N + j];
        distances[j] = diff * diff + distances[j];
      }
    }

    //Take the square root of each distance to get the euclidean distance
    for (int i = 0; i < N; i++) {
      distances[i] = sqrt(distances[i]);
    }

    //End timer for the nDimensionalEuclid function
    end = std::chrono::high_resolution_clock::now();

    //store the time it took to run the nDimensionalEuclid function
    nDimensionalEuclidTime += end - start;

    //Start timer for the classifierMergeSort function
    start = std::chrono::high_resolution_clock::now();
    
    //Parallelise merge sort the distances and the classifier arrays
    //This is a stable sort, so the classifier array will be sorted in the same way as the distances array
    mergeSort(distances, currentClassifier, 0, N - 1, temporaryOne, temporaryTwo);

    for (int i = 0; i < NC; i++) {
      classifierCount[i] = 0;
    }

    for (int i = 0; i < K; i++) {
      classifierCount[currentClassifier[i]] = classifierCount[currentClassifier[i]] + 1;
    }

    int max = 0;
    int maxIndex = 0;
    for (int i = 0; i < NC; i++) {
        if (classifierCount[i] > max) {
        max = classifierCount[i];
        maxIndex = i;
        }
    }

    //End timer for the classifierMergeSort function
    end = std::chrono::high_resolution_clock::now();

    //store the time it took to run the classifierMergeSort function
    classifierMergeSortTime += end - start;
    finalClasses[repeats] = maxIndex;
  }
  
  auto totalTime = initialisationTime + nDimensionalEuclidTime + classifierMergeSortTime;

  //output results
  fprintf(stdout, "\n||============================RESULTS============================||\n");
  fprintf(stdout, "||Initialisation Time:                                %.6f \t ||\n", initialisationTime.count());
  fprintf(stdout, "||N-Dimensional Euclid Time:                          %.6f \t ||\n", nDimensionalEuclidTime.count());
  fprintf(stdout, "||Merge Sort Time:                                    %.6f \t ||\n", classifierMergeSortTime.count());
  fprintf(stdout, "||Total Time:                                         %.6f \t ||\n", totalTime.count());
  fprintf(stdout, "||============================RESULTS============================||\n");
  for (int i = 0; i < C; i++) {
    fprintf(stdout, "||Data Point Set %d:                                     %d        ||\n", i, finalClasses[i]);
  }
  fprintf(stdout, "||============================RESULTS============================||\n");

  //Free memory space
  
  std::free(classifierCount);
  std::free(finalClasses);
  std::free(temporaryTwo);
  std::free(temporaryOne);
  std::free(classifier);
  std::free(currentClassifier);
  std::free(distances);
  std::free(queryPoint);
  std::free(nDimensionalSpace);

  return 0;
}

void checkSizes( long long &N, long long &D, long long &K, long long &C, long long &NC) {
  if (N==-1) {
    fprintf(stderr, "  No size given, using default size N = 1000\n");
    N = 1000;
  }
  if (D==-1) {
    fprintf(stderr, "  No size given, using default size D = 2\n");
    D = 2;
  }
  if (K==-1) {
    fprintf(stderr, "  No size given, using default size K = 10\n");
    K = 10;
  }
  if (C==-1) {
    fprintf(stderr, "  No size given, using default size C = 10\n");
    C = 10;
  }
  if (NC==-1) {
    fprintf(stderr, "  No size given, using default size NC = 10\n");
    NC = 10;
  }
  if (N < 1 || D < 1 || K < 1 || C < 1 || NC < 1) {
    fprintf(stderr, "Error: One or more sizes are invalid.\n");
    exit(1);
  }
  if (K > N) {
    fprintf(stderr, "Error: K cannot be greater than N.\n");
    exit(1);
  }
}

//Recursive call to merge sort
void mergeSort( float * distances, int * classifier, int lPointer, int rPointer, float * temporaryOne, int * temporaryTwo) {
  if (lPointer < rPointer) {
    int mPointer = (lPointer + rPointer) / 2;
    mergeSort(distances, classifier, lPointer, mPointer, temporaryOne, temporaryTwo);
    mergeSort(distances, classifier, mPointer + 1, rPointer, temporaryOne, temporaryTwo);
    merge(distances, classifier, lPointer, mPointer + 1, rPointer, temporaryOne, temporaryTwo);
  }
}

//Problem is that if we also want to sort the distances array, we need to pass in the classifier array as well
void merge( float * distances, int * classifier, int lPointer, int mPointer, int rPointer, float * temporaryOne, int * temporaryTwo) {
    int i = lPointer;
    int n1 = lPointer;
    int n2 = mPointer;
    while (n1 < mPointer && n2 <= rPointer) {
        if (distances[n1] < distances[n2]) {
            temporaryOne[i] = distances[n1];
            temporaryTwo[i] = classifier[n1];
            i++;
            n1++;
        }
        else {
            temporaryOne[i] = distances[n2];
            temporaryTwo[i] = classifier[n2];
            i++;
            n2++;
        }
    }
    while (n1 < mPointer) {
        temporaryOne[i] = distances[n1];
        temporaryTwo[i] = classifier[n1];
        i++;
        n1++;
    }
    while (n2 <= rPointer) {
        temporaryOne[i] = distances[n2];
        temporaryTwo[i] = classifier[n2];
        i++;
        n2++;
    }
    for (i = lPointer; i <= rPointer; i++) {
        distances[i] = temporaryOne[i];
        classifier[i] = temporaryTwo[i];
    }
}

void initialiseArrays ( float * nDimensionalSpace, int * classifier, float * queryPoint, long long &D, long long &N, long long &C, long long &NC) {
  srand(42); //Seed the random number generator for consistent results
  //Generate a random set of points to query
  for (int i = 0; i < D * C; i++) {
    queryPoint[i] = ((float) rand() / (RAND_MAX)) * 2000.0 - 1000.0;
  }

  //Fill arrays with random floats that range between -1000 and 1000
  for (int i = 0; i < D; i++) {
    for (int j = 0; j < N; j++) {
      nDimensionalSpace[ i * N + j ]  = ((float) rand() / (RAND_MAX)) * 2000.0 - 1000.0;
    }
  }
  
  //Fill array with random integers between 0 and 9 to represent our "classes"
  for (int i = 0; i < N; i++) {
      classifier[i] = ((int) rand() % NC);
  }
}

/**     1000000         10000000        100000000
 * 10       x               x               x 
 * 100      x               x               x   
 * 1000     x               x               o (not enough memory, have to use openmpi)
 * 
*/