// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The GMG V-cycle bench: the native geometric-multigrid V-cycle of
// gmg_precond.hpp, run with its AMReX kernels and with the Kokkos twins in
// kernels.hpp. Same hierarchy, same sweep counts, same control flow, same order
// of operations — only the launcher differs, which is the same discipline the
// operator bench uses.
//
// Four backends, each the previous one plus one change, so a row of the bench is
// read against the row above it:
//
//   amrex         the production per-box path — the orientation point.
//   kokkos        its per-box Kokkos twin.
//   kokkos_fused  the same kernels under one TeamPolicy launch per level.
//   kokkos_opt    ... and the halo exchange, the zero fill and the agglomeration
//                 transfers on Kokkos too (halo.hpp), which leaves no AMReX
//                 operation inside the timed cycle and therefore no reason to fence
//                 between kernels at all. The whole cycle becomes one stream the
//                 host can run ahead of.
//
// Only the Kokkos side is optimised, deliberately — the AMReX column has to stay the
// shipped V-cycle for the comparison to mean anything, and `kokkos`/`kokkos_fused`
// stay put as the intermediate baselines. Orthogonal to the backend,
// GmgArgs::agglomerate switches the hierarchy from production's in-place BoxArray
// coarsening to a re-decomposed coarse grid; since red-black smoothing is
// decomposition-independent, at equal depth that changes cost without changing a
// single arithmetic result.
//
// This is a port of the DEVICE path of GmgPrecondT<double>, reduced to what the
// timed V-cycle needs and nothing more:
//
//   kept     hierarchy construction by in-place BoxArray coarsening (so box COUNT
//            is preserved down the levels, exactly as in production, unless
//            agglomeration is asked for), RB-SOR smoothing with the reversed
//            post-sweep, fused residual+restriction,
//            piecewise-constant prolongation, the ghost fill per colour, and the
//            recursive V-cycle with warm-started sol.
//   dropped  Ginkgo (no LinOp, no Dense pack/unpack), the ReferenceExecutor host
//            path, the Chebyshev smoother and
//            its λmax power iteration, and physical boundary conditions — the
//            bench is triply periodic, so bc handling never fires and bc.hpp
//            stays out of this translation unit.
//
// The AMReX column calls the PRODUCTION kernels (gmg_kernels.hpp) rather
// than a copy of them, so the baseline is the real thing. It is recompiled here in
// the non-RDC object library, which is what makes the flags identical for both
// columns: production's _blockamr is non-RDC too (see CMakeLists.txt for the
// rationale). blockamr_kokkos stays a separate library by history, not because of an
// RDC split between the two.
//
// The templated Vcycle these backends are run through -- LevelT, sameField,
// KokkosOptGmgBackend, Precision/PrecPair -- lives in vcycle.hpp, shared with the
// production instantiation in apply.cpp; only the other three (bench-only)
// backends and the timing driver are local to this TU.

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

#include "../gmgKokkos/vcycle.hpp"

namespace blockamr::bench
{

namespace
{

// ---------------------------------------------------------------------------
// The backends. Each is the three kernels the timed V-cycle runs plus the two
// cross-runtime ordering points, and nothing else, so a backend cannot quietly
// differ in anything but what it is meant to.
//
//   afterAmrexWrite   order an AMReX write against a following backend kernel.
//   beforeAmrexRead   order a backend kernel against a following AMReX read.
//   amrexFreeCycle    the timed cycle contains no AMReX operation, so the kernels
//                     need no fence between them (they share one stream) and the
//                     data movements come from halo.hpp.
// ---------------------------------------------------------------------------

struct AmrexGmgBackend
{
    static constexpr const char* tag = "amrex";
    static constexpr bool canShareCoeffs = false;
    static constexpr bool amrexFreeCycle = false;

    // AMReX kernels are issued to AMReX's own stream, so an AMReX write before them
    // (FillBoundary, setVal) is already ordered against them: nothing to do. Nor is
    // there anything to do the other way round.
    static void afterAmrexWrite() {}

    static void beforeAmrexRead() {}

    // Not a variadic forward like the other two backends' twins below: the
    // production kernel's signature bundles its coefficients into one
    // FaceCoeffs<double>, so the flat 7-coefficient argument list vcycle.hpp's
    // shared call site passes has to be repacked here.
    static void gsColor(
        solvers::GmgFab<double>& sol,
        const solvers::GmgFab<double>& rhs,
        const solvers::GmgFab<double>& ux,
        const solvers::GmgFab<double>& lx,
        const solvers::GmgFab<double>& uy,
        const solvers::GmgFab<double>& ly,
        const solvers::GmgFab<double>& uz,
        const solvers::GmgFab<double>& lz,
        const solvers::GmgFab<double>& alpha,
        int parity,
        double omega
    )
    {
        solvers::gmgGsColorDevice<double>(
            sol,
            rhs,
            solvers::FaceCoeffs<double> {&alpha, &ux, &lx, &uy, &ly, &uz, &lz},
            parity,
            omega
        );
    }

    static void residRestrict(
        const solvers::GmgFab<double>& sol,
        const solvers::GmgFab<double>& rhs,
        solvers::GmgFab<double>& crhs,
        const solvers::GmgFab<double>& ux,
        const solvers::GmgFab<double>& lx,
        const solvers::GmgFab<double>& uy,
        const solvers::GmgFab<double>& ly,
        const solvers::GmgFab<double>& uz,
        const solvers::GmgFab<double>& lz,
        const solvers::GmgFab<double>& alpha
    )
    {
        solvers::gmgResidRestrictDevice<double>(
            sol, rhs, crhs, solvers::FaceCoeffs<double> {&alpha, &ux, &lx, &uy, &ly, &uz, &lz}
        );
    }

    template<class... A>
    static void prolongAdd(A&&... a)
    {
        solvers::gmgProlongAddDevice<double>(std::forward<A>(a)...);
    }
};

struct KokkosGmgBackend
{
    static constexpr const char* tag = "kokkos";
    static constexpr bool canShareCoeffs = false;
    static constexpr bool amrexFreeCycle = false;

    // Every kernel already fences, so a following AMReX read is ordered.
    static void beforeAmrexRead() {}

    // The other half of straddling two runtimes (the Kokkos::fence at the end of
    // each ported kernel is the first half): a Kokkos kernel about to read what
    // AMReX just wrote has no ordering against AMReX's stream, so the write has to
    // be waited on. This is the one thing the port cannot express in Kokkos, and it
    // doubles the host syncs per colour -- one for FillBoundary, one for the kernel
    // -- where the AMReX path needs only the one MFIter already performs.
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

// The same three kernels under ONE launch per level instead of one per box. Only
// the Kokkos side gets a fused variant: the AMReX column stays the production
// per-box path, so it remains the orientation point every other column is read
// against.
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
    Vcycle<Backend, T, TC> v(args);

    GmgResult r;
    r.nlevels = v.nlevels();
    r.sharedCoeffs = v.sharedCoeffs();
    r.aggLevel0 = v.aggLevel0();
    r.boxesPerLevel = v.boxesPerLevel();
    r.cellsPerLevel = v.cellsPerLevel();

    // Correctness/strength gate, untimed: how far ONE V-cycle from z0 = 0 moves the
    // residual. A launcher that indexes wrongly cannot reproduce this number.
    v.reset(*args.rhs);
    r.resid0 = std::sqrt(v.residSumSq());
    v.cycles(1);
    r.resid1 = std::sqrt(v.residSumSq());

    // Timed: every batch starts from the same state, so a batch measures
    // `iters` V-cycles of the same work rather than an ever-converging one.
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
    // Parse before the dispatch, not after: an unknown spelling must not silently
    // fall through to fp64, and silently ignoring a reduced precision on a backend
    // that has no reduced hierarchy would report an fp64 timing under its label.
    const Precision prec = parsePrecision(args.precision);
    const Precision coeffPrec = parseCoeffPrecision(args.coeffPrecision, args.precision);
    if (prec != Precision::fp64 && backend != KokkosOptGmgBackend::tag)
    {
        throw std::runtime_error(
            "benchGmgVcycle: precision '" + args.precision + "' is implemented for the '"
            + std::string(KokkosOptGmgBackend::tag) + "' backend only, not '" + backend + "'"
        );
    }
    if (coeffPrec != prec && backend != KokkosOptGmgBackend::tag)
    {
        throw std::runtime_error(
            "benchGmgVcycle: coeff_precision '" + args.coeffPrecision + "' is implemented for the '"
            + std::string(KokkosOptGmgBackend::tag) + "' backend only, not '" + backend + "'"
        );
    }
    // Same reason: a baseline silently ignoring share_coeffs would report the
    // unshared timing under a shared label.
    if (args.aggLevel0Size > 0 && backend != KokkosOptGmgBackend::tag)
    {
        throw std::runtime_error(
            "benchGmgVcycle: agg_level0_size is implemented for the '"
            + std::string(KokkosOptGmgBackend::tag) + "' backend only, not '" + backend + "'"
        );
    }
    if (args.shareCoeffs && backend != KokkosOptGmgBackend::tag)
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
            return run<B, double, solvers::Bf16>(args, iters, batches);
        case PrecPair::f32c32:
            return run<B, float>(args, iters, batches);
        case PrecPair::f32c16:
            return run<B, float, solvers::Bf16>(args, iters, batches);
        case PrecPair::f16c16:
            return run<B, solvers::Bf16>(args, iters, batches);
        case PrecPair::f64c64:
            break;
        }
        return run<B, double>(args, iters, batches);
    }
    throw std::runtime_error("benchGmgVcycle: unknown backend '" + backend + "'");
}

} // namespace blockamr::bench
