// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The GMG V-cycle bench: the native geometric-multigrid V-cycle of
// solvers/gmg_precond.hpp, run once with its AMReX kernels and once with the
// Kokkos twins in gmg_kokkos.hpp. Same hierarchy, same sweep counts, same control
// flow, same order of operations — only the launcher differs, which is the same
// discipline the operator bench uses.
//
// This is a port of the DEVICE path of GmgPrecondT<double>, reduced to what the
// timed V-cycle needs and nothing more:
//
//   kept     hierarchy construction by in-place BoxArray coarsening (so box COUNT
//            is preserved down the levels, exactly as in production), RB-SOR
//            smoothing with the reversed post-sweep, fused residual+restriction,
//            piecewise-constant prolongation, the ghost fill per colour, and the
//            recursive V-cycle with warm-started sol.
//   dropped  Ginkgo (no LinOp, no Dense pack/unpack), the ReferenceExecutor host
//            path, the FP32 hierarchy (T=double only), the Chebyshev smoother and
//            its λmax power iteration, and physical boundary conditions — the
//            bench is triply periodic, so bc handling never fires and solvers/bc
//            stays out of this translation unit.
//
// The AMReX column calls the PRODUCTION kernels (solvers/gmg_kernels.hpp) rather
// than a copy of them, so the baseline is the real thing. It is recompiled here in
// the non-RDC object library, which is what makes the flags identical for both
// columns (see CMakeLists.txt); it does mean the AMReX kernels here are non-RDC
// while production ones are RDC.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include <AMReX_MultiFab.H>
#include <AMReX_Reduce.H>

#include "../solvers/gmg_kernels.hpp"
#include "gmg_kokkos.hpp"
#include "kokkos_bench.hpp"

namespace blockamr::bench
{

namespace
{

// The level fab type production uses (FabArray<BaseFab<double>>, not MultiFab), so
// the production kernels apply unchanged.
using Fab = solvers::GmgFab<double>;

// ---------------------------------------------------------------------------
// The two backends. Each is the three kernels the timed V-cycle runs, nothing
// else, so a backend cannot quietly differ in anything but the launcher.
// ---------------------------------------------------------------------------

struct AmrexGmgBackend
{
    static constexpr const char* tag = "amrex";

    // AMReX kernels are issued to AMReX's own stream, so an AMReX write before them
    // (FillBoundary, setVal) is already ordered against them: nothing to do.
    static void afterAmrexWrite() {}

    template<class... A>
    static void gsColor(A&&... a)
    {
        solvers::gmgGsColorDevice<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void residRestrict(A&&... a)
    {
        solvers::gmgResidRestrictDevice<double>(std::forward<A>(a)...);
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

// One multigrid level, as in GmgLevelT: geometry, rediscretised coefficients and
// preallocated work fields (sol needs 1 ghost for the stencil, rhs is valid-only).
struct Level
{
    amrex::Geometry geom;
    std::unique_ptr<Fab> alpha, ux, lx, uy, ly, uz, lz, sol, rhs;
};

template<class Backend>
class Vcycle
{
public:

    Vcycle(const GmgArgs& args)
        : preSweeps_(args.preSweeps), postSweeps_(args.postSweeps),
          coarsestSweeps_(args.coarsestSweeps), omega_(args.omega)
    {
        const amrex::BoxArray& ba = args.alpha->boxArray();
        const amrex::DistributionMapping& dm = args.alpha->DistributionMap();

        levels_.push_back(makeLevel(ba, dm, *args.geom));
        solvers::gmgConvertCopyDevice(*levels_[0].alpha, *args.alpha);
        solvers::gmgConvertCopyDevice(*levels_[0].ux, *args.ux);
        solvers::gmgConvertCopyDevice(*levels_[0].lx, *args.lx);
        solvers::gmgConvertCopyDevice(*levels_[0].uy, *args.uy);
        solvers::gmgConvertCopyDevice(*levels_[0].ly, *args.ly);
        solvers::gmgConvertCopyDevice(*levels_[0].uz, *args.uz);
        solvers::gmgConvertCopyDevice(*levels_[0].lz, *args.lz);

        // Coarsen the BoxArray in place while it stays coarsenable and the coarse
        // domain keeps >= minBottom cells per direction. The DistributionMapping is
        // reused, so the number of boxes is the same on every level and only their
        // size shrinks — the production behaviour, and the reason the coarsest
        // level is launch-bound.
        while (true)
        {
            if (args.maxLevels > 0 && static_cast<int>(levels_.size()) >= args.maxLevels)
            {
                break;
            }
            const Level& f = levels_.back();
            const amrex::BoxArray& fba = f.alpha->boxArray();
            if (!fba.coarsenable(2, 2))
            {
                break;
            }
            const amrex::Box cdom = amrex::coarsen(f.geom.Domain(), 2);
            if (cdom.shortside() < args.minBottom)
            {
                break;
            }
            amrex::BoxArray cba = fba;
            cba.coarsen(2);
            const amrex::Geometry cgeom(
                cdom,
                f.geom.ProbDomain(),
                f.geom.Coord(),
                {f.geom.isPeriodic(0), f.geom.isPeriodic(1), f.geom.isPeriodic(2)}
            );
            levels_.push_back(makeLevel(cba, dm, cgeom));
            const Level& fl = levels_[levels_.size() - 2];
            Level& c = levels_.back();
            solvers::gmgRestrictDevice<double>(*fl.alpha, *c.alpha);
            solvers::gmgCoarsenFaceDevice<double>(*fl.ux, *c.ux, 0, 4.0);
            solvers::gmgCoarsenFaceDevice<double>(*fl.lx, *c.lx, 0, 4.0);
            solvers::gmgCoarsenFaceDevice<double>(*fl.uy, *c.uy, 1, 4.0);
            solvers::gmgCoarsenFaceDevice<double>(*fl.ly, *c.ly, 1, 4.0);
            solvers::gmgCoarsenFaceDevice<double>(*fl.uz, *c.uz, 2, 4.0);
            solvers::gmgCoarsenFaceDevice<double>(*fl.lz, *c.lz, 2, 4.0);
        }
        amrex::Gpu::streamSynchronize();
    }

    // L0 rhs := the caller's rhs, L0 sol := 0 — the state a preconditioner apply
    // starts from (z0 = 0, so this applies M^{-1} rather than warm-starting).
    void reset(const amrex::MultiFab& rhs)
    {
        solvers::gmgConvertCopyDevice(*levels_[0].rhs, rhs);
        levels_[0].sol->setVal(0.0);
        amrex::Gpu::streamSynchronize();
    }

    void cycles(int n)
    {
        for (int c = 0; c < n; ++c)
        {
            vcycle(0);
        }
    }

    // sum((rhs - A sol)^2) on the finest level. Reporting only: it is the gate that
    // says the timed V-cycle did the work, so it stays an AMReX reduction for both
    // backends and never runs inside a timed region.
    double residSumSq()
    {
        Level& L = levels_.front();
        fillGhosts(L);
        const auto psi = L.sol->const_arrays();
        const auto b = L.rhs->const_arrays();
        const auto ax = L.ux->const_arrays();
        const auto lxa = L.lx->const_arrays();
        const auto ay = L.uy->const_arrays();
        const auto lya = L.ly->const_arrays();
        const auto az = L.uz->const_arrays();
        const auto lza = L.lz->const_arrays();
        const auto al = L.alpha->const_arrays();
        return amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpSum> {},
            amrex::TypeList<double> {},
            *L.rhs,
            amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) -> amrex::GpuTuple<double>
            {
                const double aE = ax[box](i + 1, j, k);
                const double aW = lxa[box](i, j, k);
                const double aN = ay[box](i, j + 1, k);
                const double aS = lya[box](i, j, k);
                const double aT = az[box](i, j, k + 1);
                const double aB = lza[box](i, j, k);
                const double off = aE * psi[box](i + 1, j, k) + aW * psi[box](i - 1, j, k)
                                 + aN * psi[box](i, j + 1, k) + aS * psi[box](i, j - 1, k)
                                 + aT * psi[box](i, j, k + 1) + aB * psi[box](i, j, k - 1);
                const double diag = al[box](i, j, k) - (aE + aW + aN + aS + aT + aB);
                const double r = b[box](i, j, k) - (diag * psi[box](i, j, k) + off);
                return {r * r};
            }
        );
    }

    int nlevels() const { return static_cast<int>(levels_.size()); }

    // Boxes and cells PER LEVEL: the point of the exercise is that the box count
    // does not shrink while the cell count does.
    std::vector<int> boxesPerLevel() const
    {
        std::vector<int> v;
        for (const Level& L : levels_)
        {
            v.push_back(static_cast<int>(L.alpha->boxArray().size()));
        }
        return v;
    }

    std::vector<long> cellsPerLevel() const
    {
        std::vector<long> v;
        for (const Level& L : levels_)
        {
            v.push_back(static_cast<long>(L.alpha->boxArray().numPts()));
        }
        return v;
    }

private:

    static std::unique_ptr<Fab>
    makeMf(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int ng)
    {
        auto mf = std::make_unique<Fab>(ba, dm, 1, ng);
        mf->setVal(0.0);
        return mf;
    }

    static Level makeLevel(
        const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const amrex::Geometry& geom
    )
    {
        Level L;
        L.geom = geom;
        L.alpha = makeMf(ba, dm, 0);
        const auto fba = [&ba](int d)
        { return amrex::convert(ba, amrex::IntVect::TheDimensionVector(d)); };
        L.ux = makeMf(fba(0), dm, 0);
        L.lx = makeMf(fba(0), dm, 0);
        L.uy = makeMf(fba(1), dm, 0);
        L.ly = makeMf(fba(1), dm, 0);
        L.uz = makeMf(fba(2), dm, 0);
        L.lz = makeMf(fba(2), dm, 0);
        L.sol = makeMf(ba, dm, 1);
        L.rhs = makeMf(ba, dm, 0);
        return L;
    }

    // Periodic/internal ghosts only — the bench mesh is triply periodic, so the
    // physical-BC reflection of the production fillGhosts has nothing to do. Stays
    // AMReX for both backends: it is a halo exchange, not a cell kernel.
    void fillGhosts(Level& L) const
    {
        L.sol->FillBoundary(L.geom.periodicity());
        Backend::afterAmrexWrite();
    }

    // Red-black colour sweeps; `reversed` flips the colour order (black-red), the
    // adjoint of the forward sweep, so the whole V-cycle stays symmetric.
    void smooth(std::size_t l, int sweeps, bool reversed)
    {
        Level& L = levels_[l];
        for (int s = 0; s < sweeps; ++s)
        {
            for (int c = 0; c < 2; ++c)
            {
                const int parity = (reversed ? 1 + c : c) & 1;
                fillGhosts(L); // the other colour changed — refresh ghosts
                Backend::gsColor(
                    *L.sol,
                    *L.rhs,
                    *L.ux,
                    *L.lx,
                    *L.uy,
                    *L.ly,
                    *L.uz,
                    *L.lz,
                    *L.alpha,
                    parity,
                    omega_
                );
            }
        }
    }

    void vcycle(std::size_t l)
    {
        Level& L = levels_[l];
        if (l + 1 == levels_.size())
        {
            // Tiny grid: forward + reversed halves keep the coarsest "solve"
            // self-adjoint.
            smooth(l, coarsestSweeps_ / 2, false);
            smooth(l, coarsestSweeps_ / 2, true);
            return;
        }
        smooth(l, preSweeps_, false);
        fillGhosts(L);
        Level& C = levels_[l + 1];
        Backend::residRestrict(
            *L.sol, *L.rhs, *C.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha
        );
        C.sol->setVal(0.0);
        Backend::afterAmrexWrite();
        vcycle(l + 1);
        Backend::prolongAdd(*C.sol, *L.sol);
        smooth(l, postSweeps_, true);
    }

    int preSweeps_;
    int postSweeps_;
    int coarsestSweeps_;
    double omega_;
    std::vector<Level> levels_;
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

template<class Backend>
GmgResult run(const GmgArgs& args, int iters, int batches)
{
    Vcycle<Backend> v(args);

    GmgResult r;
    r.nlevels = v.nlevels();
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
    return {AmrexGmgBackend::tag, KokkosGmgBackend::tag};
}

GmgResult benchGmgVcycle(const std::string& backend, const GmgArgs& args, int iters, int batches)
{
    if (backend == AmrexGmgBackend::tag)
    {
        return run<AmrexGmgBackend>(args, iters, batches);
    }
    if (backend == KokkosGmgBackend::tag)
    {
        return run<KokkosGmgBackend>(args, iters, batches);
    }
    throw std::runtime_error("benchGmgVcycle: unknown backend '" + backend + "'");
}

} // namespace blockamr::bench
