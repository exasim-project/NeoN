// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/executor/serialExecutor.hpp"
#include "NeoN/core/memory/kokkos.hpp"

NeoN::SerialExecutor::SerialExecutor() : SerialExecutor(std::make_unique<DefaultAllocator>()) {};

NeoN::SerialExecutor::SerialExecutor(std::unique_ptr<AllocatorStrategy> strategy)
{
    strategy->setMemSpace(MemorySpace::CPU);
    allocContext_ = std::make_shared<AllocatorContext>();
    allocContext_->setStrategy(std::move(strategy));
}

NeoN::SerialExecutor::~SerialExecutor() {};
