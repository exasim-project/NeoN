// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/executor/GPUExecutor.hpp"
#include "NeoN/core/memory/kokkos.hpp"

NeoN::GPUExecutor::GPUExecutor() : GPUExecutor(std::make_unique<DefaultAllocator>()) {};

NeoN::GPUExecutor::GPUExecutor(std::unique_ptr<AllocatorStrategy> strategy)
{
    strategy->setMemSpace(MemorySpace::GPU);
    allocContext_ = std::make_shared<AllocatorContext>();
    allocContext_->setStrategy(std::move(strategy));
}

NeoN::GPUExecutor::~GPUExecutor() {};
