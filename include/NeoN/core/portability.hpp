// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

/* @file portability.hpp
 * @brief Kokkos-free function annotations shared by device and host-only translation units.
 *
 * NeoN compiles in two halves (see doc/splitCompilation.rst):
 *   - device TUs include Kokkos, may launch kernels (NEON_LAMBDA, parallelFor) and are built with
 *     the device toolchain at C++20;
 *   - host-only TUs never see a Kokkos header and may be built with a different host compiler and
 *     a newer standard (NeoN_HOST_CXX_STANDARD). They are compiled with NEON_HOST_ONLY_TU=1.
 *
 * Headers shared by both halves must therefore not include Kokkos. Functions that have to be
 * callable from kernels use NEON_INLINE_FUNCTION from this header instead of
 * KOKKOS_INLINE_FUNCTION. The expansion matches Kokkos' own for the CUDA and HIP backends; for all
 * other backends (Serial, OpenMP, Threads, SYCL) Kokkos expands it to plain `inline` as well.
 *
 * The annotation differs between a CUDA/HIP TU and a host-only TU. That affects neither layout nor
 * mangling of the annotated function, and it is the same arrangement Eigen and Thrust rely on for
 * mixed .cpp/.cu builds.
 */

#include <cstdio>
#include <cstdlib>

// Device printf is only declared by the HIP runtime header, not by <cstdio>.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NEON_HOST_DEVICE __host__ __device__
#else
#define NEON_HOST_DEVICE
#endif

#define NEON_INLINE_FUNCTION NEON_HOST_DEVICE inline

namespace NeoN::detail
{

/* @brief Abort the program from host or device code.
 *
 * Replacement for Kokkos::abort in headers that must stay Kokkos-free. On CUDA and HIP devices the
 * message is printed and the kernel traps; on SYCL devices the kernel traps without a message,
 * since device printf needs the SYCL headers; on the host it is printed to stderr before
 * std::abort.
 */
NEON_INLINE_FUNCTION void abort(const char* message)
{
#if defined(__CUDA_ARCH__)
    printf("NeoN abort: %s\n", message);
    __trap();
#elif defined(__HIP_DEVICE_COMPILE__)
    printf("NeoN abort: %s\n", message);
    __builtin_trap();
#elif defined(__SYCL_DEVICE_ONLY__)
    (void)message;
    __builtin_trap();
#else
    std::fprintf(stderr, "NeoN abort: %s\n", message);
    std::abort();
#endif
}

} // namespace NeoN::detail

#define NEON_ABORT(message) ::NeoN::detail::abort(message)
