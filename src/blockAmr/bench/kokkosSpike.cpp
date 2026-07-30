// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Feasibility spike for the operator bench: a Kokkos CUDA kernel links and launches from inside
// _blockamr, and an unmanaged Kokkos View over MultiFab memory is a no-copy handle on the same
// bytes AMReX kernels see (MultiFab data is plain unmanaged device memory).

#include <stdexcept>

#include <Kokkos_Core.hpp>

#include <AMReX_MultiFab.H>

#include "NeoN/blockAmr/bench/kokkosBench.hpp"

namespace blockamr
{

namespace
{

// LayoutLeft matches Array4's i-contiguous ordering, so this View addresses the fab as
// amrex::ParallelFor does.
using FabView = Kokkos::View<
    double***,
    Kokkos::LayoutLeft,
    Kokkos::DefaultExecutionSpace::memory_space,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

void requireKokkos()
{
    if (!Kokkos::is_initialized())
    {
        throw std::runtime_error(
            "Kokkos is not initialized -- open a blockamr.runtime() first. (Kokkos cannot be "
            "re-initialized after finalize, so a second sequential runtime() block in the same "
            "process leaves it unavailable.)"
        );
    }
}

} // namespace

std::string kokkosExecutionSpace() { return std::string(Kokkos::DefaultExecutionSpace::name()); }

double kokkosSelftest(long n)
{
    requireKokkos();
    double sum = 0.0;
    Kokkos::parallel_reduce(
        "selftest",
        Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(const long i, double& acc) { acc += static_cast<double>(i); },
        sum
    );
    Kokkos::fence();
    return sum;
}

double kokkosMfSum(amrex::MultiFab& mf)
{
    requireKokkos();
    double total = 0.0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& fbx = mf[mfi].box();
        const amrex::Box& vbx = mfi.validbox();
        // The View spans the whole fab box, so valid indices shift by validbox - fabbox smallEnd.
        FabView v(mf[mfi].dataPtr(), fbx.length(0), fbx.length(1), fbx.length(2));
        const int ox = vbx.smallEnd(0) - fbx.smallEnd(0);
        const int oy = vbx.smallEnd(1) - fbx.smallEnd(1);
        const int oz = vbx.smallEnd(2) - fbx.smallEnd(2);

        double sum = 0.0;
        Kokkos::parallel_reduce(
            "mf_sum",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0, 0, 0}, {vbx.length(0), vbx.length(1), vbx.length(2)}
            ),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& acc) {
                acc += v(i + ox, j + oy, k + oz);
            },
            sum
        );
        Kokkos::fence();
        total += sum;
    }
    return total;
}

} // namespace blockamr
