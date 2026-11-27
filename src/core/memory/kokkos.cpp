// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/memory/kokkos.hpp"

#include <cstdlib>
#include <memory>

#include "NeoN/core/error.hpp"

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
        // unlike kokkos_malloc, std::malloc reports a failure only through its result
        void* ptr = std::malloc(size);
        NF_ASSERT(ptr != nullptr || size == 0, "Host allocation failed");
        return ptr;
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
        // on failure std::realloc returns null and keeps the original allocation
        void* newPtr = std::realloc(ptr, size);
        NF_ASSERT(newPtr != nullptr || size == 0, "Host reallocation failed");
        return newPtr;
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
