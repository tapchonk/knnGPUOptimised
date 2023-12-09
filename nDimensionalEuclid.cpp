#include <cstdlib>
#include <utility>
#include <stdlib.h>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "nDimensionalEuclid.h"

#ifdef KOKKOS_ENABLE_CUDA
    #define CurrentExecutionSpace Kokkos::CudaSpace::execution_space
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    #define CurrentExecutionSpace Kokkos::OpenMP::execution_space
#endif
/**
 * @brief Calculates the euclidean distance between a query point and a set of points in an n-dimensional space.
 * 
 * The multithreading is done by partitioning the nDimensional Space partitions into smaller partitions for each worker thread to work on.
 * The access patterns depend on whether we want row order or column order dependent on the hardware we detect or the devices the user specifies with the KOKKOS_DEVICES environment variable.
 * When compiling for CPUs, we use the FORTRAN style of column order where the left most index denotes contiguous access as we iterate over dimensions to avoid race conditions.
 * For GPUs, we take advantage of the fact that we have stacked HBM2 memory or GDDR6 memory and so row order access is optimal in this case as GPU threads are grouped into warps.
 * Otherwise, for GPUs we can just redefine the access pattern to be column order for older GPUs that do not have stacked memory chips.
 * The definition of the MemLayout is defined in the main file, where in my have Kokkos::LayoutRight (row order) or Kokkos::LayoutLeft (column order) access.
 * 
 * @param nDimensionalSpace The view representing the n-dimensional space containing the set of points (NDimensionalSpaceView)
 * @param queryPoint The view representing the query point (QueryPointView)
 * @param distances The view representing the distances between the query point and the set of points (DistanceView)
 * @param nDimensionalPartition The view representing the partition of the n-dimensional space (NDimensionalPartitionView)
 * @param N The number of points in the n-dimensional space (long long)
 * @param D The dimensionality of the n-dimensional space (long long)
 * @param C The number of repeats (long long)
 * @param partitions The number of partitions (long long)
 * @param partitionD The size of each partition (long long)
 * 
 * @return void
 */
void nDimensionalEuclid(  NDimensionalSpaceView nDimensionalSpace, QueryPointView queryPoint, DistanceView distances, NDimensionalPartitionView nDimensionalPartition,
                          long long N, long long D, long long C, long long partitions, long long partitionD ) {
    //Defines the range policy for the parallel for loops, the execution space is defined at the top of the file dependent on Kokkos device configuration        
    using current_range_policy = Kokkos::RangePolicy<CurrentExecutionSpace, Kokkos::Schedule<Kokkos::Static>>;
    
    for (long long workloadPartitioning = 0; workloadPartitioning < partitions; workloadPartitioning++) {
        long long endIndex  = partitionD;
        long long toAdd = workloadPartitioning * partitionD;
        if (workloadPartitioning == (partitions - 1) && D % partitionD != 0) {
            endIndex = D % partitionD;
        }
        //if the GPU doesn't have enough memory to store the entire n-dimensional space, we need to copy the partition to the GPU for every iteration of the workload loop
        #ifdef KOKKOS_ENABLE_CUDA
        if (workloadPartitioning != 0) {
            auto currWorkload = Kokkos::subview(nDimensionalSpace, workloadPartitioning, Kokkos::ALL(), Kokkos::ALL());
            Kokkos::deep_copy(nDimensionalPartition, currWorkload);
        }
        #endif
        //repeat for each query point we would like to classify
        for (long long repeats = 0; repeats < C; repeats++) {
            //note to self: loop unrolling does nothing in this case as vectorisation is achieved implcitly as "warps" on GPUs
            Kokkos::parallel_for( "n_dimensional_euclid_calc",  current_range_policy(0, N), KOKKOS_LAMBDA (long long i) {
                float diff = 0.0f;
                float diffSquare = 0.0f;
                for (long long j = 0; j < endIndex; j++) {
                    diffSquare = queryPoint( repeats, toAdd + j ) - nDimensionalPartition(  j, i );
                    diffSquare *= diffSquare;
                    diff += diffSquare;
                }
                distances( repeats, i ) += diff;
            });
        }
    }
    
    //Take the square root of each distance to get the euclidean distance
    //This may actually be pointless since sqrt is a decreasing function and we are sorting by non-decreasing order
    Kokkos::parallel_for("n_dimensional_euclid_root", current_range_policy(0, C * N), KOKKOS_LAMBDA (int i) {
        distances( i/N, i%N ) = sqrt(distances( i/N, i%N ));
    });

    Kokkos::fence();
    
    return;
}

