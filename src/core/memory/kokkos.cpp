// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/memory/kokkos.hpp"

namespace NeoN
{

void* KokkosAllocator::alloc(size_t size)
{
    switch (memSpace_)
    {
    case MemorySpace::GPU:
    {
        return Kokkos::kokkos_malloc<GPUMemSpace>("Vector", size);
    }
    case MemorySpace::CPU:
    {
        return Kokkos::kokkos_malloc<CPUMemSpace>("Vector", size);
    }
    default:
        NF_ERROR_EXIT("Unknown memory space");
    }
}

void* KokkosAllocator::realloc(void* ptr, size_t size)
{
    switch (memSpace_)
    {
    case MemorySpace::GPU:
    {
        return Kokkos::kokkos_realloc<GPUMemSpace>(ptr, size);
    }
    case MemorySpace::CPU:
    {
        return Kokkos::kokkos_realloc<CPUMemSpace>(ptr, size);
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
        Kokkos::kokkos_free<CPUMemSpace>(ptr);
        break;
    }
    default:
        NF_ERROR_EXIT("Unknown memory space");
    }
}

}
