#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include "checkSizes.h"


/**
 * @brief Error checks the sizes given by the user.
 * 
 * @param N The number of data points (long long)
 * @param D The number of dimensions (long long)
 * @param K The number of nearest neighbours (long long)
 * @param C The number of query points we would like to classify (long long)
 * @param NC The number of unique classes given for our algorithm to classify the data amongst (long long)
 * 
 * @return exit(-1) if one or more sizes are invalid.
 */
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
    exit(-1);
  }
  if (K > N) {
    fprintf(stderr, "Error: K cannot be greater than N.\n");
    exit(-1);
  }
}