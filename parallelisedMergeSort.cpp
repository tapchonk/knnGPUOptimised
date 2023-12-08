#include <cstdlib>
#include <vector>
#include <utility>
#include <stdlib.h>
#include <Kokkos_Core.hpp>

#include "parallelisedMergeSort.h"

#ifdef KOKKOS_ENABLE_OPENMP
using openmp_range_policy = Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static>>;

//Problem is that if we also want to sort the distances array, we need to pass in the classifier array as well
void merge(  DistanceView distances, ClassifierView classifier, int lPointer, int mPointer, int rPointer,  TemporaryOneView temporaryOne, TemporaryTwoView temporaryTwo, int dataSetIndex ) {
    int i = lPointer;
    int n1 = lPointer;
    int n2 = mPointer;
    while (n1 < mPointer && n2 <= rPointer) {
        if (distances(dataSetIndex, n1) < distances(dataSetIndex, n2)) {
            temporaryOne(i) = distances(dataSetIndex, n1);
            temporaryTwo(i) = classifier(dataSetIndex, n1);
            i++;
            n1++;
        }
        else {
            temporaryOne(i) = distances(dataSetIndex, n2);
            temporaryTwo(i) = classifier(dataSetIndex, n2);
            i++;
            n2++;
        }
    }
    while (n1 < mPointer) {
        temporaryOne(i) = distances(dataSetIndex, n1);
        temporaryTwo(i) = classifier(dataSetIndex, n1);
        i++;
        n1++;
    }
    while (n2 <= rPointer) {
        temporaryOne(i) = distances(dataSetIndex, n2);
        temporaryTwo(i) = classifier(dataSetIndex, n2);
        i++;
        n2++;
    }
    for (i = lPointer; i <= rPointer; i++) {
        distances(dataSetIndex, i) = temporaryOne(i);
        classifier(dataSetIndex, i) = temporaryTwo(i);
    }
}

void mergeSort(  DistanceView distances, ClassifierView classifier, int lPointer, int rPointer, TemporaryOneView temporaryOne, TemporaryTwoView temporaryTwo, int dataSetIndex ) {
    if (lPointer < rPointer) {
        int mPointer = (lPointer + rPointer) / 2;
        mergeSort(distances, classifier, lPointer, mPointer, temporaryOne, temporaryTwo, dataSetIndex);
        mergeSort(distances, classifier, mPointer + 1, rPointer, temporaryOne, temporaryTwo, dataSetIndex);
        merge(distances, classifier, lPointer, mPointer + 1, rPointer, temporaryOne, temporaryTwo, dataSetIndex);
    }
}

/**
 * @brief Performs parallelised merge sort on the given data.
 *
 * This function splits the workload of merge sort across multiple threads using OpenMP as a backend.
 * It sorts the 'distances' and 'classifier' arrays based on the values in the 'distances' array.
 * The implementation divides the workload amongst equal partitions to sort each partition in parallel.
 * We first sort each subarray in parallel, then merge the subarrays together also in parallel.
 * 
 * @param distances The 2D view of distances between data points. Each row represents a data set. 
 * @param classifier The 2D view of classifiers corresponding to the data points. Each row represents a data set.
 * @param N The number of data points in each data set.
 * @param temporaryOne The temporary view used for merging the 'distances' array.
 * @param temporaryTwo The temporary view used for merging the 'classifier' array.
 * @param repeats The index of the current data set being sorted.
 * @param numThreads The number of threads to be used for parallelisation.
 * 
 * @return void
 */
void parallelisedMergeSort(  DistanceView distances, ClassifierView classifier, long long N, TemporaryOneView temporaryOne, TemporaryTwoView temporaryTwo, int repeats, int numThreads ) {
    Kokkos::parallel_for( "parallelise_merge_sort", openmp_range_policy(0, numThreads), KOKKOS_LAMBDA ( int i ) {
        if(i == numThreads - 1)
            mergeSort(distances, classifier, ((N - 1)/numThreads)*i, N - 1, temporaryOne, temporaryTwo, repeats);
        else
            mergeSort(distances, classifier, ((N - 1)/numThreads)*i, ((N - 1)/numThreads) * (i + 1) - 1, temporaryOne, temporaryTwo, repeats);
    });

    int i = 0;
    std::vector<int> mergePoints;
    bool isSorted = false;
    mergePoints.push_back(0);
    for (i = 0; i < N; ++i) {
        if (distances( repeats, i ) > distances( repeats, i+1 )) {
            mergePoints.push_back(i);
            mergePoints.push_back(i+1);
        }
    }
    while (!isSorted) {
        if (mergePoints.size()<4) {
            isSorted = true;
            continue;
        }

        Kokkos::parallel_for( "merging", openmp_range_policy(0, mergePoints.size()/4), KOKKOS_LAMBDA ( int i ) {
            merge(distances, classifier, mergePoints[i * 4], mergePoints[i * 4 + 2], mergePoints[i * 4 + 3], temporaryOne, temporaryTwo, repeats);
        });
        for (i = 1; i < mergePoints.size() - 2; i+=2) {
            mergePoints.erase(mergePoints.begin() + i);
            mergePoints.erase(mergePoints.begin() + i);
        }
    }

    return;
}

#endif