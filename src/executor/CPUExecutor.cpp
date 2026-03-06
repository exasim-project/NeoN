// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/executor/CPUExecutor.hpp"
#include "NeoN/core/memory/kokkos.hpp"

NeoN::CPUExecutor::CPUExecutor() : CPUExecutor(std::make_unique<DefaultAllocator>()) {};

NeoN::CPUExecutor::CPUExecutor(std::unique_ptr<AllocatorStrategy> strategy)
{
    strategy->setMemSpace(MemorySpace::CPU);
    allocContext_ = std::make_shared<AllocatorContext>();
    allocContext_->setStrategy(std::move(strategy));
}

NeoN::CPUExecutor::~CPUExecutor() {};
