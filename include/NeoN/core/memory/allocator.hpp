// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/error.hpp"

#include <memory>


namespace NeoN
{


enum class MemorySpace
{
    CPU,
    GPU
};


class AllocatorStrategy
{

protected:

    // TODO check if enforcing via Ctor is an option
    MemorySpace memSpace_ = MemorySpace::CPU;

public:

    /**@brief set tag controlling where allocation happens*/
    void setMemSpace(MemorySpace memSpace) { memSpace_ = memSpace; }

    /**@brief allocate a block of size bytes */
    virtual void* alloc(size_t size) = 0;

    /**@brief reallocate a block of size bytes */
    virtual void* realloc(void* ptr, size_t size) = 0;

    /**@brief free a block of size bytes */
    virtual void free(void* ptr) = 0;

    virtual ~AllocatorStrategy() {}
};


class AllocatorContext
{

    std::unique_ptr<AllocatorStrategy> strategy {};

public:

    void setStrategy(std::unique_ptr<AllocatorStrategy> strategyIn)
    {
        strategy = std::move(strategyIn);
    }

    /**@brief allocate a space for n elements of type T  */
    template<typename T>
    T* alloc(size_t elements) const
    {
        if (!strategy)
        {
            NF_ERROR_EXIT("No allocator set");
        }
        return reinterpret_cast<T*>(strategy->alloc(sizeof(T) * elements));
    }

    /**@brief reallocate a space for n elements of type T  */
    template<typename T>
    T* realloc(void* ptr, size_t elements) const
    {
        if (!strategy)
        {
            NF_ERROR_EXIT("No allocator set");
        }
        return reinterpret_cast<T*>(strategy->realloc(ptr, sizeof(T) * elements));
    }

    void free(void* ptr) const
    {
        if (!strategy)
        {
            NF_ERROR_EXIT("No allocator set");
        }
        return strategy->free(ptr);
    }
};


}
