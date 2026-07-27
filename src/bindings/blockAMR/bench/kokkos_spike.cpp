// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Feasibility spike for the Kokkos-vs-AMReX operator bench. Two things need to
// hold before the bench is worth writing:
//
//   1. a Kokkos CUDA kernel links and launches from inside _blockamr;
//   2. an unmanaged Kokkos View over MultiFab memory is a valid handle on the
//      same bytes AMReX kernels see -- MultiFab data is plain device memory
//      (the_arena_is_managed defaults to false), so DefaultExecutionSpace's
//      memory space is the correct label and no copy is involved.
//
// This TU is compiled WITHOUT relocatable device code, like the rest of the module
// (see CMakeLists.txt for why _blockamr itself is also non-RDC).

#include <stdexcept>

#include <Kokkos_Core.hpp>

#include <AMReX_MultiFab.H>

#include "kokkos_bench.hpp"
#include "launch.hpp"

namespace blockamr::bench
{

namespace
{

// LayoutLeft matches Array4's i-contiguous ordering, so this View addresses the
// fab exactly as amrex::ParallelFor does.
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

void kokkosInitialize()
{
    if (!Kokkos::is_initialized() && !Kokkos::is_finalized())
    {
        Kokkos::initialize(Kokkos::InitializationSettings());
    }
}

void kokkosFinalize()
{
    if (Kokkos::is_initialized())
    {
        // The kokkos_stream backend's execution space instances own cudaStreams
        // (ManageStream::yes), so they have to go before finalize -- and before
        // amrex::Finalize tears the CUDA context down.
        releaseStreamPool();
        Kokkos::finalize();
    }
}

bool kokkosInitialized() { return Kokkos::is_initialized(); }

bool kokkosFinalized() { return Kokkos::is_finalized(); }

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
        // The View spans the whole fab box, ghosts included, so valid-region
        // indices are offset by validbox.smallEnd - fabbox.smallEnd.
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

} // namespace blockamr::bench
