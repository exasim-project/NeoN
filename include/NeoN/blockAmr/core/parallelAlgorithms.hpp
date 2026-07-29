// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <variant>

#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_MultiFab.H>

#include "NeoN/core/executor/executor.hpp"

// One lambda form for all launchers: __host__ __device__ satisfies both an AMReX
// device lambda and a Kokkos functor, so the kernel body is written once.
#define BLOCKAMR_LAMBDA [=] AMREX_GPU_HOST_DEVICE

namespace blockamr
{

namespace detail
{

// Every executor alternative forwards to amrex::ParallelFor: the seam, not a backend
// change -- codegen must stay identical, so swapping an arm to Kokkos later is one
// visible edit here and nowhere else.

template<typename ExecutorType, class Kernel>
void parallelForImpl(
    const ExecutorType&,
    const amrex::Box& bx,
    Kernel kernel,
    [[maybe_unused]] const std::string& name
)
{
    amrex::ParallelFor(bx, kernel);
}

template<typename ExecutorType, class Kernel>
void parallelForImpl(
    const ExecutorType&,
    const amrex::MultiFab& mf,
    Kernel kernel,
    [[maybe_unused]] const std::string& name
)
{
    amrex::ParallelFor(mf, kernel);
}

} // namespace detail

// Per-box: kernel(i, j, k) over the box's GLOBAL index range.
template<class Kernel>
void parallelFor(
    const NeoN::Executor& exec,
    const amrex::Box& bx,
    Kernel kernel,
    std::string name = "parallelFor"
)
{
    std::visit([&](const auto& e) { detail::parallelForImpl(e, bx, kernel, name); }, exec);
}

// Fused: kernel(ibox, i, j, k) over every valid cell of every box in ONE launch.
template<class Kernel>
void parallelFor(
    const NeoN::Executor& exec,
    const amrex::MultiFab& mf,
    Kernel kernel,
    std::string name = "parallelFor"
)
{
    std::visit([&](const auto& e) { detail::parallelForImpl(e, mf, kernel, name); }, exec);
}

} // namespace blockamr
