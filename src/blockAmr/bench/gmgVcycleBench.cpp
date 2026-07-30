// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The GMG V-cycle bench: gmgPrecond.hpp's V-cycle over four launchers, each the previous one
// plus one change. The Vcycle template lives in vcycle.hpp, shared with production's apply.cpp.
// Backends and port scope: report/blockamr-linear-algebra-notes.md#gmg-v-cycle-bench-backends

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include <AMReX_MultiFab.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>

#include "NeoN/blockAmr/bench/kokkosBench.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/vcycle.hpp"

namespace blockamr
{

namespace
{

// A backend is the three kernels of the timed V-cycle plus two ordering points: afterAmrexWrite
// orders an AMReX write against a following backend kernel, beforeAmrexRead the reverse.
// amrexFreeCycle means no AMReX operation inside the cycle, hence no fence between the kernels.

struct AmrexGmgBackend
{
    static constexpr const char* tag = "amrex";
    static constexpr bool canShareCoeffs = false;
    static constexpr bool amrexFreeCycle = false;

    // Same stream as the AMReX writes, so both directions are already ordered.
    static void afterAmrexWrite() {}

    static void beforeAmrexRead() {}

    // Not a variadic forward: the production kernel takes one FaceCoeffs<double>, so the flat
    // argument list of vcycle.hpp's shared call site is repacked here.
    static void gsColor(
        la::GmgFab<double>& sol,
        const la::GmgFab<double>& rhs,
        const la::GmgFab<double>& ux,
        const la::GmgFab<double>& lx,
        const la::GmgFab<double>& uy,
        const la::GmgFab<double>& ly,
        const la::GmgFab<double>& uz,
        const la::GmgFab<double>& lz,
        const la::GmgFab<double>& alpha,
        int parity,
        double omega
    )
    {
        la::gmgGsColor<double>(
            sol,
            rhs,
            la::FaceCoeffs<double> {&alpha, &ux, &lx, &uy, &ly, &uz, &lz},
            parity,
            omega,
            /*onDevice=*/true
        );
    }

    static void residRestrict(
        const la::GmgFab<double>& sol,
        const la::GmgFab<double>& rhs,
        la::GmgFab<double>& crhs,
        const la::GmgFab<double>& ux,
        const la::GmgFab<double>& lx,
        const la::GmgFab<double>& uy,
        const la::GmgFab<double>& ly,
        const la::GmgFab<double>& uz,
        const la::GmgFab<double>& lz,
        const la::GmgFab<double>& alpha
    )
    {
        la::gmgResidRestrict<double>(
            sol,
            rhs,
            crhs,
            la::FaceCoeffs<double> {&alpha, &ux, &lx, &uy, &ly, &uz, &lz},
            /*onDevice=*/true
        );
    }

    template<class... A>
    static void prolongAdd(A&&... a)
    {
        la::gmgProlongAdd<double>(std::forward<A>(a)..., /*onDevice=*/true);
    }
};

struct KokkosGmgBackend
{
    static constexpr const char* tag = "kokkos";
    static constexpr bool canShareCoeffs = false;
    static constexpr bool amrexFreeCycle = false;

    // Every kernel already fences, so a following AMReX read is ordered.
    static void beforeAmrexRead() {}

    // A Kokkos kernel has no ordering against AMReX's stream, so an AMReX write before it has to
    // be waited on: two host syncs per colour, where the AMReX path needs only MFIter's one.
    static void afterAmrexWrite() { amrex::Gpu::streamSynchronizeAll(); }

    template<class... A>
    static void gsColor(A&&... a)
    {
        gmgGsColorKokkos<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void residRestrict(A&&... a)
    {
        gmgResidRestrictKokkos<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void prolongAdd(A&&... a)
    {
        gmgProlongAddKokkos<double>(std::forward<A>(a)...);
    }
};

// The same three kernels under ONE launch per level instead of one per box.
struct KokkosFusedGmgBackend
{
    static constexpr const char* tag = "kokkos_fused";
    static constexpr bool canShareCoeffs = false;
    static constexpr bool amrexFreeCycle = false;

    static void beforeAmrexRead() {}

    static void afterAmrexWrite() { amrex::Gpu::streamSynchronizeAll(); }

    template<class... A>
    static void gsColor(A&&... a)
    {
        gmgGsColorKokkosFused<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void residRestrict(A&&... a)
    {
        gmgResidRestrictKokkosFused<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void prolongAdd(A&&... a)
    {
        gmgProlongAddKokkosFused<double>(std::forward<A>(a)...);
    }
};

// Fence both runtimes regardless of backend, as in benchOperator.
void fenceAll()
{
    amrex::Gpu::streamSynchronize();
    if (Kokkos::is_initialized())
    {
        Kokkos::fence();
    }
}

template<class Backend, class T, class TC = T>
GmgResult run(const GmgArgs& args, int iters, int batches)
{
    Vcycle<Backend, T, TC> v(
        *args.geom,
        *args.alpha,
        *args.ux,
        *args.lx,
        *args.uy,
        *args.ly,
        *args.uz,
        *args.lz,
        args.opts
    );

    GmgResult r;
    r.nlevels = v.nlevels();
    r.sharedCoeffs = v.sharedCoeffs();
    r.aggLevel0 = v.aggLevel0();
    r.boxesPerLevel = v.boxesPerLevel();
    r.cellsPerLevel = v.cellsPerLevel();

    // Untimed strength gate: how far ONE V-cycle from z0 = 0 moves the residual. A launcher that
    // indexes wrongly cannot reproduce this number.
    v.reset(*args.rhs);
    r.resid0 = std::sqrt(v.residSumSq());
    v.cycles(1);
    r.resid1 = std::sqrt(v.residSumSq());

    // Timed: every batch restarts from the same state, so each measures the same work.
    v.reset(*args.rhs);
    v.cycles(1);
    fenceAll();

    std::vector<double> ms, msEnq;
    ms.reserve(static_cast<std::size_t>(batches));
    msEnq.reserve(static_cast<std::size_t>(batches));
    for (int b = 0; b < batches; ++b)
    {
        v.reset(*args.rhs);
        fenceAll();
        const auto t0 = std::chrono::steady_clock::now();
        v.cycles(iters);
        const auto t1 = std::chrono::steady_clock::now();
        fenceAll();
        const auto t2 = std::chrono::steady_clock::now();
        msEnq.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
        ms.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count() / iters);
    }
    std::sort(ms.begin(), ms.end());
    std::sort(msEnq.begin(), msEnq.end());
    r.msMin = ms.front();
    r.msMedian = ms[ms.size() / 2];
    r.msEnqueue = msEnq.front();
    return r;
}

} // namespace

std::vector<std::string> benchGmgBackends()
{
    return {
        AmrexGmgBackend::tag,
        KokkosGmgBackend::tag,
        KokkosFusedGmgBackend::tag,
        KokkosOptGmgBackend::tag
    };
}

GmgResult benchGmgVcycle(const std::string& backend, const GmgArgs& args, int iters, int batches)
{
    // Parsed before the dispatch: the baselines stay fp64 by refusal, not by ignoring a reduced
    // precision they have no hierarchy for and reporting an fp64 timing under its label.
    const Precision prec = parsePrecision(args.opts.precision);
    const Precision coeffPrec = parseCoeffPrecision(args.opts.coeffPrecision, args.opts.precision);
    if (prec != Precision::fp64 && backend != KokkosOptGmgBackend::tag)
    {
        throw std::runtime_error(
            "benchGmgVcycle: precision '" + args.opts.precision + "' is implemented for the '"
            + std::string(KokkosOptGmgBackend::tag) + "' backend only, not '" + backend + "'"
        );
    }
    if (coeffPrec != prec && backend != KokkosOptGmgBackend::tag)
    {
        throw std::runtime_error(
            "benchGmgVcycle: coeff_precision '" + args.opts.coeffPrecision
            + "' is implemented for the '" + std::string(KokkosOptGmgBackend::tag)
            + "' backend only, not '" + backend + "'"
        );
    }
    // Same reason: a baseline ignoring these would report its timing under another label.
    if (args.opts.aggLevel0Size > 0 && backend != KokkosOptGmgBackend::tag)
    {
        throw std::runtime_error(
            "benchGmgVcycle: agg_level0_size is implemented for the '"
            + std::string(KokkosOptGmgBackend::tag) + "' backend only, not '" + backend + "'"
        );
    }
    if (args.opts.shareCoeffs && backend != KokkosOptGmgBackend::tag)
    {
        throw std::runtime_error(
            "benchGmgVcycle: share_coeffs is implemented for the '"
            + std::string(KokkosOptGmgBackend::tag) + "' backend only, not '" + backend + "'"
        );
    }
    if (backend == AmrexGmgBackend::tag)
    {
        return run<AmrexGmgBackend, double>(args, iters, batches);
    }
    if (backend == KokkosGmgBackend::tag)
    {
        return run<KokkosGmgBackend, double>(args, iters, batches);
    }
    if (backend == KokkosFusedGmgBackend::tag)
    {
        return run<KokkosFusedGmgBackend, double>(args, iters, batches);
    }
    if (backend == KokkosOptGmgBackend::tag)
    {
        using B = KokkosOptGmgBackend;
        switch (precPair(prec, coeffPrec))
        {
        case PrecPair::f64c32:
            return run<B, double, float>(args, iters, batches);
        case PrecPair::f64c16:
            return run<B, double, la::Bf16>(args, iters, batches);
        case PrecPair::f32c32:
            return run<B, float>(args, iters, batches);
        case PrecPair::f32c16:
            return run<B, float, la::Bf16>(args, iters, batches);
        case PrecPair::f16c16:
            return run<B, la::Bf16>(args, iters, batches);
        case PrecPair::f64c64:
            break;
        }
        return run<B, double>(args, iters, batches);
    }
    throw std::runtime_error("benchGmgVcycle: unknown backend '" + backend + "'");
}

} // namespace blockamr
