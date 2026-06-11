// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/memory/allocator.hpp"

#if NF_WITH_UMPIRE
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/ResourceManager.hpp"
#endif

namespace NeoN
{

#if NF_WITH_UMPIRE
class UmpireMempoolHandler
{
private:

    static bool hasPool(MemorySpace memSpace);

public:

    static void setupUmpirePool(MemorySpace memSpace, size_t size);

    static auto getUmpirePool(MemorySpace memSpace)
    {
        auto& rm = umpire::ResourceManager::getInstance();
        if (memSpace == MemorySpace::GPU)
        {
            return rm.getAllocator("DEVICE_POOL");
        }
        return rm.getAllocator("HOST_POOL");
    }

    static void destroyUmpirePool(MemorySpace memSpace)
    {
        auto pool = getUmpirePool(memSpace);
        pool.release();
    }
};
#endif


class UmpireAllocator : public AllocatorStrategy
{

public:

    void* alloc(size_t size) override;

    void* realloc(void* ptr, size_t size) override;

    void free(void* ptr) override;

    ~UmpireAllocator() override {}
};


class UmpirePoolAllocator : public AllocatorStrategy
{

public:

    void* alloc(size_t size) override;

    void* realloc(void* ptr, size_t size) override;

    void free(void* ptr) override;

    ~UmpirePoolAllocator() override {}
};


}
