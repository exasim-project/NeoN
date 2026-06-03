// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/memory/allocator.hpp"

#if NF_WITH_UMPIRE
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/ResourceManager.hpp"
#endif

#include <cstddef>
#include <string>

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

    /** @brief Bytes the pool currently reserves from the device — its real ceiling.
     *  0 if the pool does not exist.
     *
     *  Note: a QuickPool only ever returns *whole, completely-free* blocks to the device
     *  (and only via an explicit release, not implemented here). It cannot hand back the
     *  unused space inside a block that still holds any live allocation, so a pool
     *  pre-sized to the device maximum stays fully reserved for the whole run. These
     *  accessors exist to *report* that ceiling, not to change it — to fit more cells
     *  the peak working set itself must shrink. */
    static std::size_t actualSize(MemorySpace memSpace);

    /** @brief Bytes currently handed out (in use) from the pool. 0 if no pool. */
    static std::size_t currentSize(MemorySpace memSpace);

    /** @brief High-water mark of in-use bytes over the pool's lifetime. 0 if no pool. */
    static std::size_t highWatermark(MemorySpace memSpace);

    /** @brief Log in-use / reserved / high-water pool sizes (MB) at info level so the
     *  device-memory ceiling that limits cells-per-GPU is visible during a run. */
    static void logStats(MemorySpace memSpace, const std::string& label = "");

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
