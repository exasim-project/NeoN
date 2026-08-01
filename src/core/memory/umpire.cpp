// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_UMPIRE
#include "umpire/Allocator.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/ResourceManager.hpp"
#endif

#include "NeoN/core/memory/umpire.hpp"


namespace NeoN
{

#if NF_WITH_UMPIRE

bool UmpireMempoolHandler::hasPool(MemorySpace memSpace)
{
    auto& rm = umpire::ResourceManager::getInstance();
    auto names = rm.getAllocatorNames();
    auto name = (memSpace == MemorySpace::GPU) ? "DEVICE_POOL" : "HOST_POOL";

    for (auto& n : names)
    {
        if (name == n)
        {
            return true;
        }
    }
    return false;
}

void UmpireMempoolHandler::setupUmpirePool(MemorySpace memSpace, size_t size)
{
    auto& rm = umpire::ResourceManager::getInstance();

    // do nothing if pool already exists
    if (hasPool(memSpace)) return;

    switch (memSpace)
    {
    case MemorySpace::GPU:
    {
        auto allocator = rm.getAllocator("DEVICE");
        rm.makeAllocator<umpire::strategy::QuickPool>("DEVICE_POOL", allocator, size);
        break;
    }
    case MemorySpace::CPU:
    {
        auto allocator = rm.getAllocator("HOST");
        rm.makeAllocator<umpire::strategy::QuickPool>("HOST_POOL", allocator, size);
        break;
    }
    default:
        NF_ERROR_EXIT("Unknown memory space");
    }
}


void* UmpireAllocator::alloc(size_t size)
{
    auto& rm = umpire::ResourceManager::getInstance();
    switch (memSpace_)
    {
    case MemorySpace::GPU:
    {
        auto allocator = rm.getAllocator("DEVICE");
        return allocator.allocate(size);
    }
    case MemorySpace::CPU:
    {
        auto allocator = rm.getAllocator("HOST");
        return allocator.allocate(size);
    }
    default:
        NF_ERROR_EXIT("Unknown memory space");
    }
    return nullptr;
}

void* UmpireAllocator::realloc(void* ptr, size_t size)
{
    auto& rm = umpire::ResourceManager::getInstance();
    return rm.reallocate(ptr, size);
}

void UmpireAllocator::free(void* ptr)
{
    auto& rm = umpire::ResourceManager::getInstance();

    if (memSpace_ == MemorySpace::GPU)
    {
        auto allocator = rm.getAllocator("DEVICE");
        allocator.deallocate(ptr);
    }
    else if (memSpace_ == MemorySpace::CPU)
    {
        auto allocator = rm.getAllocator("HOST");
        allocator.deallocate(ptr);
    }
}

void* UmpirePoolAllocator::alloc(size_t size)
{
    auto allocator = UmpireMempoolHandler::getUmpirePool(memSpace_);
    return allocator.allocate(size);
}

void* UmpirePoolAllocator::realloc(void* ptr, size_t size)
{
    auto& rm = umpire::ResourceManager::getInstance();
    return rm.reallocate(ptr, size);
}

void UmpirePoolAllocator::free(void* ptr)
{
    auto allocator = UmpireMempoolHandler::getUmpirePool(memSpace_);
    allocator.deallocate(ptr);
}

#else

void* UmpireAllocator::alloc(size_t size)
{
    NF_ERROR_EXIT("Not implemented. Build NeoN with -DNeoN_WITH_UMPIRE");
    return nullptr;
}

void* UmpireAllocator::realloc(void* ptr, size_t size)
{
    NF_ERROR_EXIT("Not implemented. Build NeoN with -DNeoN_WITH_UMPIRE");
    return nullptr;
}

void UmpireAllocator::free(void* ptr)
{
    NF_ERROR_EXIT("Not implemented. Build NeoN with -DNeoN_WITH_UMPIRE");
}

void* UmpirePoolAllocator::alloc(size_t size)
{
    NF_ERROR_EXIT("Not implemented. Build NeoN with -DNeoN_WITH_UMPIRE");
}

void* UmpirePoolAllocator::realloc(void* ptr, size_t size)
{
    NF_ERROR_EXIT("Not implemented. Build NeoN with -DNeoN_WITH_UMPIRE");
    return nullptr;
}

void UmpirePoolAllocator::free(void* ptr)
{
    NF_ERROR_EXIT("Not implemented. Build NeoN with -DNeoN_WITH_UMPIRE");
}
#endif

}
