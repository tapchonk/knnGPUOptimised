 *
 * ************************************************************************
 *
 *                        Kokkos v. 4.0
 *       Copyright (2022) National Technology  Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
 * See https://kokkos.org/LICENSE for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 

 # Kokkos Optimized KNN Algorithm
 
 This algorithm implements the K-nearest neighbors (KNN) algorithm using Kokkos, a programming model for performance portability. 
 It is optimized for both Nvidia GPUs and CPUs using OpenMP.
 
 To run the algorithm, follow these steps:
 
 ## 1. Install CUDA 11.2 and GCC 9.2.0 on your system.
 
 ## 2. Clone the Kokkos GitHub repository by executing the following command in your terminal:
    ```
    git clone https://github.com/kokkos/kokkos.git kokkos-master
    ```
 
 ## 3. Ensure that "kokkos-master" is in your home directory.
    ```
    cd ~/kokkos-master
    ```
 
 ## 4. Compile the code using the following command for Nvidia GPUs:
    ```
    make -j KOKKOS_DEVICES=Cuda
    ```
    or the following command for CPUs:
    ```
    make -j KOKKOS_DEVICES=OpenMP
    ```
 
 ## 5. Run the application locally using the following command for Nvidia GPUs:
    ```
    ./knnGPUOptimised.cuda -N 1000 -D 2 -K 100 -NC 10 -C 20
    ```
    or the following command for CPUs:
    ```
    ./knnGPUOptimised.host -N 1000 -D 2 -K 100 -NC 10 -C 20
    ```
 
 ## 6. To run the application on a batch compute system, prepend `./remoterun.sh` to the above commands.
    Ensure that the script has been updated for the batch compute and for that partition.
    Also ensure that you specify if you want the Cuda-optimised version or the OpenMP-optimised version in the remote run script.
 
 For more information and updates, please visit the Kokkos GitHub repository: [kokkos/kokkos](https://github.com/kokkos/kokkos)

