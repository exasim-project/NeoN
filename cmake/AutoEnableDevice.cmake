# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

message(STATUS "Auto detecting accelerator devices")
include(CheckLanguage)

if(NeoN_WITH_OMP AND NeoN_WITH_THREADS)
  message(FATAL_ERROR "NeoN_WITH_OMP and NeoN_WITH_THREADS are mutally exclusive")
endif()

if(NeoN_WITH_OMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    message(STATUS "Set Kokkos_ENABLE_OPENMP=ON")
    set(Kokkos_ENABLE_OPENMP
        ON
        CACHE INTERNAL "")
    set(Kokkos_ENABLE_THREADS
        OFF
        CACHE INTERNAL "")
  endif()
elseif(NeoN_WITH_THREADS)
  find_package(Threads QUIET)
  if(Threads_FOUND)
    message(STATUS "Set Kokkos_ENABLE_Threads=ON")
    set(Kokkos_ENABLE_THREADS
        ON
        CACHE INTERNAL "")
    set(Kokkos_ENABLE_OPENMP
        OFF
        CACHE INTERNAL "")
  endif()
else()
  set(Kokkos_ENABLE_THREADS
      OFF
      CACHE INTERNAL "")
  set(Kokkos_ENABLE_OPENMP
      OFF
      CACHE INTERNAL "")
endif()

if(NOT DEFINED Kokkos_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Only enable CUDA as a first-class CMake language when Kokkos compiles as a CMake language. In
    # traditional mode (nvcc_wrapper) Kokkos enables it itself and injects --expt-extended-lambda;
    # enabling it here first suppresses that.
    if(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
      enable_language(CUDA)
    endif()
    set(NeoN_ENABLE_CUDA
        ON
        CACHE INTERNAL "")
    message(STATUS "Set Kokkos_ENABLE_CUDA=ON")
    set(Kokkos_ENABLE_CUDA
        ON
        CACHE INTERNAL "")
  else()
    set(NeoN_ENABLE_CUDA
        OFF
        CACHE INTERNAL "")

    set(Kokkos_ENABLE_CUDA
        OFF
        CACHE INTERNAL "")
  endif()
else()
  message(STATUS "Skip CUDA detection Kokkos_ENABLE_CUDA=${Kokkos_ENABLE_CUDA}")
  if(Kokkos_ENABLE_CUDA)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
      message(FATAL_ERROR "Kokkos_ENABLE_CUDA=ON but no CUDA compiler was found")
    endif()
    # See note above: only take over the CUDA language in cmake-language mode.
    if(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
      enable_language(CUDA)
    endif()
    set(NeoN_ENABLE_CUDA
        ON
        CACHE INTERNAL "")
    set(Kokkos_ENABLE_CUDA_CONSTEXPR
        ON
        CACHE INTERNAL "")
  else()
    set(NeoN_ENABLE_CUDA
        OFF
        CACHE INTERNAL "")
  endif()
endif()

# Kokkos_ENABLE_CUDA_CONSTEXPR must be ON for any CUDA build, NOT only the auto-detected one. It is
# what makes Kokkos pass --expt-relaxed-constexpr to nvcc. Without it, NeoN device kernels that
# index a View (a constexpr operator[] forwarding to std::span) hit nvcc warning 20013 ("constexpr
# __host__ function called from __host__ __device__ function") and are SILENTLY miscompiled -- the
# device-side element writes simply vanish, so GPU results are garbage while serial/host results are
# correct. When CUDA is enabled explicitly via -DKokkos_ENABLE_CUDA=ON the detection branch above is
# skipped, so force it here as well.
if(Kokkos_ENABLE_CUDA)
  set(Kokkos_ENABLE_CUDA_CONSTEXPR
      ON
      CACHE INTERNAL "" FORCE)
endif()

if(NOT DEFINED Kokkos_ENABLE_HIP)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    message(STATUS "Set Kokkos_ENABLE_HIP=ON")
    set(Kokkos_ENABLE_HIP
        ON
        CACHE INTERNAL "")
  else()
    set(Kokkos_ENABLE_HIP
        OFF
        CACHE INTERNAL "")
  endif()
else()
  message(STATUS "Skip HIP detection Kokkos_ENABLE_HIP=${Kokkos_ENABLE_HIP}")
endif()

if(NOT DEFINED Kokkos_ENABLE_SERIAL)
  set(Kokkos_ENABLE_SERIAL
      ON
      CACHE INTERNAL "")
endif()
