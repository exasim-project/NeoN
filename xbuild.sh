#!/bin/bash
module purge
module load llvm/19.1.7
# module load cuda/12.5.1
module load cmake
module load openmpi
#
# rm -rf build
#
# export NEON_DEVICE=nvidia_h200
#

    # -DCMAKE_CUDA_ARCHITECTURES=native \
    # -DCMAKE_CUDA_HOST_COMPILER=clang++ \
    # -DCMAKE_CUDA_COMPILER=nvcc \
    # -DKokkos_ENABLE_CUDA=ON \
# CC=clang CXX=clang++ cmake --preset julia \
#     -DNeoN_WITH_THREADS=OFF \
#     -DNeoN_WITH_UMPIRE=OFF \
#     -DNeoN_WITH_OMP=OFF \
#     -DNeoN_WITH_GINKGO=OFF \
#     -DNeoN_WITH_CUDA=ON \
#     -DKokkos_ENABLE_HIP=OFF \

# CPU Multithread
# CC=clang CXX=clang++ 
# time (
cmake --preset julia \
		-DNeoN_WITH_GINKGO=OFF \
		-DKokkos_ENABLE_ATOMICS_BYPASS=OFF \
		-DNeoN_WITH_JULIA=ON \
		-DNeoN_WITH_CUDA=OFF \
		-DNeoN_WITH_MPI=OFF \
		-DNeoN_WITH_OMP=ON \
	    -DKokkos_ENABLE_OPENMP=ON \
		-DNeoN_WITH_THREADS=OFF \
	    -DKokkos_ENABLE_SERIAL=ON \
        -DKokkos_EMABLE_CUDA=OFF \
	    -DKokkos_ENABLE_THREADS=OFF;
# 	cmake --build --preset julia --parallel 64
# )	

# H200 GPU
# CC=clang CXX=clang++ cmake --preset julia \
# 		-DNeoN_WITH_GINKGO=OFF \
# 		-DKokkos_ENABLE_ATOMICS_BYPASS=OFF \
# 		-DNeoN_WITH_JULIA=ON \
# 		-DKokkos_ARCH_HOPPER90=ON \
# 		-DCMAKE_CUDA_ARCHITECTURES=90 \
# 		-DNeoN_WITH_MPI=OFF \
# 		-DNeoN_WITH_CUDA=ON \
# 	    -DKokkos_ENABLE_SERIAL=ON \
# 	    -DKokkos_ENABLE_OPENMP=OFF \
# 	    -DKokkos_ENABLE_THREADS=OFF \
# 		-DNeoN_WITH_UMPIRE=OFF


# time (CC=clang CXX=clang++ cmake --preset julia \
# 		-DNeoN_WITH_GINKGO=ON \
# 		-DKokkos_ENABLE_ATOMICS_BYPASS=OFF \
# 		-DNeoN_WITH_JULIA=OFF \
# 		-DKokkos_ARCH_HOPPER90=ON \
# 		-DCMAKE_CUDA_ARCHITECTURES=90 \
# 		-DNeoN_WITH_CUDA=ON \
# 	    -DKokkos_ENABLE_SERIAL=ON \
# 		-DNeoN_WITH_MPI=OFF \
# 	    -DKokkos_ENABLE_OPENMP=OFF \
# 	    -DKokkos_ENABLE_THREADS=OFF \
# 		-DNeoN_WITH_UMPIRE=OFF;
# )
cmake --build --preset julia --parallel 32