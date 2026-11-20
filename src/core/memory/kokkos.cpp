// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/memory/kokkos.hpp"

#include <cstdlib>

namespace NeoN
{

// Host memory is taken from malloc instead of Kokkos::HostSpace: HostSpace::deallocate calls
// Kokkos::fence() (all execution spaces, i.e. a device synchronisation) before every free, and
// host vectors (e.g. the boundary offsets of every field) are freed many times per time step.
// Host memory of the serial and CPU executors is never accessed by device kernels, and host
// execution is synchronous, so the fence protects nothing here.

void* KokkosAllocator::alloc(sizeType size)
{
    switch (memSpace_)
    {
    case MemorySpace::GPU:
    {
        return Kokkos::kokkos_malloc<GPUMemSpace>("Vector", size);
    }
    case MemorySpace::CPU:
    {
        return std::malloc(size);
    }
    default:
        NF_ERROR_EXIT("Unknown memory space");
    }
}

void* KokkosAllocator::realloc(void* ptr, sizeType size)
{
    switch (memSpace_)
    {
    case MemorySpace::GPU:
    {
        return Kokkos::kokkos_realloc<GPUMemSpace>(ptr, size);
    }
    case MemorySpace::CPU:
    {
        return std::realloc(ptr, size);
    }
    default:
        NF_ERROR_EXIT("Unknown memory space");
    }
}

void KokkosAllocator::free(void* ptr)
{
    switch (memSpace_)
    {
    case MemorySpace::GPU:
    {
        Kokkos::kokkos_free<GPUMemSpace>(ptr);
        break;
    }
    case MemorySpace::CPU:
    {
        std::free(ptr);
        break;
    }
    default:
        NF_ERROR_EXIT("Unknown memory space");
    }
}

}
