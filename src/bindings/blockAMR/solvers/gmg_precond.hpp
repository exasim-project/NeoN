// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Arena.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bc.hpp"
#include "gmg_kernels.hpp"
#include "linop_base.hpp"
#include "profiling.hpp"
#include "transfer.hpp"
#include "types.hpp"

namespace blockamr::solvers
{

// One multigrid level: geometry, rediscretised coefficients and preallocated
// work fields (sol needs 1 ghost for the stencil; rhs is valid-only).
template<class T>
struct GmgLevelT
{
    amrex::Geometry geom;
    std::shared_ptr<GmgFab<T>> alpha, ux, lx, uy, ly, uz, lz;
    std::shared_ptr<GmgFab<T>> sol, rhs;
    std::shared_ptr<GmgFab<T>> chebD; // Chebyshev increment (only when smoother="chebyshev")
    double lambdaMax = 0.0;           // estimate of lambda_max(D^{-1}A) on this level
};

// Native matrix-free geometric-multigrid V-cycle preconditioner on the
// face-coefficient operator: z = M^{-1} r via `n_cycles` V-cycles with
// red-black Gauss-Seidel smoothing (the same smoother family MLMG uses;
// measured much stronger than damped Jacobi here: 9/9 vs 16/16 CG iterations
// at N=32/64 with omega=6/7 Jacobi, 20/22 with omega=2/3), volume-average
// restriction and piecewise-constant prolongation. The V-cycle is symmetric —
// the post-smoother runs the colours in REVERSED order (black-red), making it
// the adjoint of the pre-smoother, and prolongation is the adjoint of
// restriction up to a constant — so it is CG-safe. The whole hierarchy is
// built ONCE at construction — no per-apply allocation; the coefficients are
// copied, so later in-place updates to the caller's fields are seen by the
// outer operator but not by this preconditioner (a slightly stale
// preconditioner only costs iterations).
// Abstract hook exposing a GMG V-cycle as operations on FP64 MultiFabs, so the
// native stationary solver (FaceCoeffSolver solver="gmg") can drive the
// precision-templated GmgPrecondT<T> without knowing T. The whole apply runs on
// AMReX fabs (no Ginkgo vector), converting FP64<->T at the two ends. M3 fuses the
// FP64 residual, its convert-scatter into the (T-typed) L0 rhs and the FP64 norm
// into one kernel (residScatterNorm); vcycleGather runs the V-cycle(s) and adds
// the correction back onto the FP64 x.
class GmgApplyMf
{
public:

    virtual ~GmgApplyMf() = default;

    // Fused r = rhs - A*sol - shift -> (cast to T) L0 rhs; L0 sol := 0; returns the
    // FP64 sum of squares of r (norm authority stays double even for a float L0
    // rhs). `sol`'s ghosts must already be filled by the caller.
    virtual double residScatterNorm(
        const amrex::MultiFab& sol,
        const amrex::MultiFab& rhs,
        const amrex::MultiFab& ux,
        const amrex::MultiFab& lx,
        const amrex::MultiFab& uy,
        const amrex::MultiFab& ly,
        const amrex::MultiFab& uz,
        const amrex::MultiFab& lz,
        const amrex::MultiFab& alpha,
        double shift
    ) const = 0;

    // Run nCycles_ V-cycles on the L0 rhs set by residScatterNorm, then x += the
    // (converted) L0 correction.
    virtual void vcycleGather(amrex::MultiFab& x) const = 0;
};

template<class T>
class GmgPrecondT : public AmrexLinOpBase<GmgPrecondT<T>>, public GmgApplyMf
{
public:

    explicit GmgPrecondT(std::shared_ptr<const gko::Executor> exec)
        : AmrexLinOpBase<GmgPrecondT<T>>(exec)
    {}

    GmgPrecondT(
        std::shared_ptr<const gko::Executor> exec,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        amrex::Geometry geom,
        gko::size_type n,
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        BcArray bc,
        int n_cycles,
        int pre_sweeps,
        int post_sweeps,
        int coarsest_sweeps,
        int max_levels,
        int min_bottom,
        const std::string& smoother
    )
        : AmrexLinOpBase<GmgPrecondT<T>>(exec, gko::dim<2> {n, n}), bc_(bc),
          hasPhysBc_(std::any_of(bc.begin(), bc.end(), [](int b) { return b != 0; })),
          onDevice_(exec->get_master().get() != exec.get()), nCycles_(n_cycles),
          preSweeps_(pre_sweeps), postSweeps_(post_sweeps), coarsestSweeps_(coarsest_sweeps),
          useCheb_(smoother == "chebyshev")
    {
        if (smoother != "rbgs" && smoother != "chebyshev")
        {
            throw std::runtime_error(
                "GmgPrecond: unknown gmg_smoother '" + smoother
                + "' (expected 'rbgs' or 'chebyshev')"
            );
        }
        // Finest level: copy the coefficients into this preconditioner's arena
        // (default/device on cuda, pinned on reference — MultiFab::Copy handles
        // the cross-arena transfer, cf. pinnedCopy).
        levels_.push_back(makeLevel(ba, dm, geom));
        copyCoeff(*levels_[0].alpha, *alpha);
        copyCoeff(*levels_[0].ux, *ux);
        copyCoeff(*levels_[0].lx, *lx);
        copyCoeff(*levels_[0].uy, *uy);
        copyCoeff(*levels_[0].ly, *ly);
        copyCoeff(*levels_[0].uz, *uz);
        copyCoeff(*levels_[0].lz, *lz);

        // Coarsen by 2 while every box dimension stays divisible by 2 (with
        // >= 2 cells left) and the coarse domain keeps >= 4 cells per
        // direction. The coarse coefficients are rediscretised from the fine
        // ones: face coeff = mean of the 4 covered fine face coeffs / 4
        // (a ~ -beta/dx^2: beta averaged, dx doubled), alpha (per-volume
        // source) = mean of the 8 fine cell values.
        while (true)
        {
            if (max_levels > 0 && static_cast<int>(levels_.size()) >= max_levels)
            {
                break;
            }
            const GmgLevelT<T>& f = levels_.back();
            const amrex::BoxArray& fba = f.alpha->boxArray();
            if (!fba.coarsenable(2, 2))
            {
                break;
            }
            const amrex::Box cdom = amrex::coarsen(f.geom.Domain(), 2);
            if (cdom.shortside() < min_bottom)
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
            GmgLevelT<T>& c = levels_.back();
            const GmgLevelT<T>& fl = levels_[levels_.size() - 2];
            if (onDevice_)
            {
                gmgRestrictDevice(*fl.alpha, *c.alpha);
                gmgCoarsenFaceDevice(*fl.ux, *c.ux, 0, 4.0);
                gmgCoarsenFaceDevice(*fl.lx, *c.lx, 0, 4.0);
                gmgCoarsenFaceDevice(*fl.uy, *c.uy, 1, 4.0);
                gmgCoarsenFaceDevice(*fl.ly, *c.ly, 1, 4.0);
                gmgCoarsenFaceDevice(*fl.uz, *c.uz, 2, 4.0);
                gmgCoarsenFaceDevice(*fl.lz, *c.lz, 2, 4.0);
            }
            else
            {
                gmgRestrictHost(*fl.alpha, *c.alpha);
                gmgCoarsenFaceHost(*fl.ux, *c.ux, 0, 4.0);
                gmgCoarsenFaceHost(*fl.lx, *c.lx, 0, 4.0);
                gmgCoarsenFaceHost(*fl.uy, *c.uy, 1, 4.0);
                gmgCoarsenFaceHost(*fl.ly, *c.ly, 1, 4.0);
                gmgCoarsenFaceHost(*fl.uz, *c.uz, 2, 4.0);
                gmgCoarsenFaceHost(*fl.lz, *c.lz, 2, 4.0);
            }
        }
        amrex::Gpu::streamSynchronize();

        // Chebyshev setup: per level allocate the polynomial increment field and
        // estimate lambda_max(D^{-1}A) via ~15 power iterations (setup-time cost).
        if (useCheb_)
        {
            for (auto& L : levels_)
            {
                L.chebD = makeMf(L.alpha->boxArray(), L.alpha->DistributionMap(), 0);
            }
            for (std::size_t l = 0; l < levels_.size(); ++l)
            {
                levels_[l].lambdaMax = estimateLambdaMax(l);
            }
            amrex::Gpu::streamSynchronize();
        }
    }

    // Native stationary-solver hooks (M1 + M3). residScatterNorm forms the FP64
    // residual and, in the SAME kernel, casts it into the T-typed L0 rhs and
    // reduces its FP64 norm — no separate FP64 residual MultiFab, norm pass, or
    // convert-scatter. vcycleGather then runs the V-cycle(s) and adds the T-typed
    // correction back onto the FP64 x. Runs entirely on AMReX fabs (no Ginkgo
    // vector); conversions are identities when T==double.
    double residScatterNorm(
        const amrex::MultiFab& sol,
        const amrex::MultiFab& rhs,
        const amrex::MultiFab& ux,
        const amrex::MultiFab& lx,
        const amrex::MultiFab& uy,
        const amrex::MultiFab& ly,
        const amrex::MultiFab& uz,
        const amrex::MultiFab& lz,
        const amrex::MultiFab& alpha,
        double shift
    ) const override
    {
        const GmgLevelT<T>& L0 = levels_.front();
        double sumsq;
        if (onDevice_)
        {
            sumsq = faceCoeffResidScatterNormDevice<T>(
                sol, rhs, ux, lx, uy, ly, uz, lz, alpha, shift, *L0.rhs
            );
            L0.sol->setVal(T(0)); // z0 = 0: apply M^{-1}, not a warm-started solve
        }
        else
        {
            sumsq = faceCoeffResidScatterNormHost<T>(
                sol, rhs, ux, lx, uy, ly, uz, lz, alpha, shift, *L0.rhs
            );
            L0.sol->setVal(T(0));
            amrex::Gpu::streamSynchronize();
        }
        return sumsq;
    }

    void vcycleGather(amrex::MultiFab& x) const override
    {
        const GmgLevelT<T>& L0 = levels_.front();
        if (onDevice_)
        {
            {
                prof::Timer t("gmg.vcycle");
                for (int c = 0; c < nCycles_; ++c)
                {
                    vcycle(0);
                }
            }
            {
                prof::Timer t("gmg.solve.gather");
                gmgConvertAddDevice(x, *L0.sol); // x += (double) L0 correction
                amrex::Gpu::streamSynchronize();
            }
        }
        else
        {
            for (int c = 0; c < nCycles_; ++c)
            {
                vcycle(0);
            }
            gmgConvertAddHost(x, *L0.sol);
            amrex::Gpu::streamSynchronize();
        }
    }

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) visible in this
    // scope (the declaration below would otherwise hide it).
    using AmrexLinOpBase<GmgPrecondT<T>>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto exec = this->get_executor();
        const GmgLevelT<T>& L0 = levels_.front();
        if (onDevice_)
        {
            prof::Timer tAll("gmg.apply");
            {
                prof::Timer t("gmg.sync_gko");
                exec->synchronize(); // b written by Ginkgo
            }
            {
                prof::Timer t("gmg.scatter");
                scatter_device(gko::as<Dense>(b)->get_const_values(), *L0.rhs);
                L0.sol->setVal(0.0); // z0 = 0: apply M^{-1}, not a warm-started solve
            }
            {
                prof::Timer t("gmg.vcycle");
                for (int c = 0; c < nCycles_; ++c)
                {
                    vcycle(0);
                }
            }
            {
                prof::Timer t("gmg.gather");
                gather_device(*L0.sol, gko::as<Dense>(x)->get_values(), 1.0);
                amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
            }
        }
        else
        {
            auto host = exec->get_master();
            auto bHost = gko::clone(host, gko::as<Dense>(b));
            scatter(bHost->get_const_values(), *L0.rhs);
            L0.sol->setVal(0.0);
            amrex::Gpu::streamSynchronize(); // setVal may run on the GPU stream
            for (int c = 0; c < nCycles_; ++c)
            {
                vcycle(0);
            }
            auto xHost = Dense::create(host, gko::as<Dense>(x)->get_size());
            gather(*L0.sol, xHost->get_values(), 1.0);
            gko::as<Dense>(x)->copy_from(xHost);
        }
    }

private:

    // Chebyshev smooths modes with eigenvalue in [lambdaMax / kChebEigRatio,
    // lambdaMax]; the lower modes are left to the coarse grid. alpha ~= 4-8 is
    // the usual band; 6 minimised the CG count here (degree-2 -> 11 iters at
    // N=32/64 vs rbgs 9, a sweep over {2,3,4,6,8,15,30} at setup).
    static constexpr double kChebEigRatio = 6.0;
    static constexpr double kChebSafety = 1.05; // inflate the lambda_max estimate
    static constexpr int kPowerIters = 15;      // power iterations for lambda_max

    std::shared_ptr<GmgFab<T>>
    makeMf(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int ng) const
    {
        auto mf = onDevice_ ? std::make_shared<GmgFab<T>>(ba, dm, 1, ng)
                            : std::make_shared<GmgFab<T>>(
                                ba, dm, 1, ng, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
                            );
        mf->setVal(T(0));
        return mf;
    }

    GmgLevelT<T> makeLevel(
        const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const amrex::Geometry& geom
    ) const
    {
        GmgLevelT<T> L;
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

    // Copy the caller's FP64 coefficient MultiFab into a level fab, converting
    // to T. On the reference path the source may live in device memory, so it is
    // staged through a pinned FP64 copy before the host conversion loop.
    void copyCoeff(GmgFab<T>& dst, const amrex::MultiFab& src) const
    {
        if (onDevice_)
        {
            gmgConvertCopyDevice(dst, src);
        }
        else
        {
            auto tmp = pinnedCopy(src);
            amrex::Gpu::streamSynchronize();
            gmgConvertCopyHost(dst, *tmp);
        }
    }

    // Fill sol's ghost layer: periodic/internal via FillBoundary, then the
    // homogeneous Dirichlet/Neumann reflection on domain faces (the gap-2 BC
    // fills coarsen cleanly, so the same bc spec applies on every level).
    void fillGhosts(const GmgLevelT<T>& L, int lvl) const
    {
        prof::Timer t("gmg.fill", lvl);
        L.sol->FillBoundary(L.geom.periodicity());
        if (!onDevice_)
        {
            amrex::Gpu::streamSynchronize(); // FillBoundary before host loops
        }
        if (hasPhysBc_)
        {
            if (onDevice_)
            {
                fillDomainBcGhostsDevice(*L.sol, L.geom.Domain(), bc_);
            }
            else
            {
                fillDomainBcGhostsHost(*L.sol, L.geom.Domain(), bc_);
            }
        }
    }

    // Dispatch to the configured smoother. `reversed` is only meaningful for
    // red-black Gauss-Seidel (post-smoother runs the colours in reversed order,
    // the adjoint of the forward sweep); Chebyshev is symmetric by construction
    // so it ignores it. `sweeps` is the RB-GS sweep count / the Chebyshev degree.
    void smooth(std::size_t l, int sweeps, bool reversed) const
    {
        if (useCheb_)
        {
            chebyshevSmooth(l, sweeps);
        }
        else
        {
            rbgsSmooth(l, sweeps, reversed);
        }
    }

    // Red-black Gauss-Seidel sweeps; `reversed` flips the colour order
    // (black-red), which is the adjoint of the forward sweep — used for the
    // post-smoother so the whole V-cycle is symmetric.
    void rbgsSmooth(std::size_t l, int sweeps, bool reversed) const
    {
        const GmgLevelT<T>& L = levels_[l];
        for (int s = 0; s < sweeps; ++s)
        {
            for (int c = 0; c < 2; ++c)
            {
                const int parity = (reversed ? 1 + c : c) & 1;
                fillGhosts(L, static_cast<int>(l)); // the other colour changed — refresh ghosts
                if (onDevice_)
                {
                    prof::Timer t("gmg.gs", static_cast<int>(l));
                    gmgGsColorDevice(
                        *L.sol, *L.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha, parity
                    );
                }
                else
                {
                    gmgGsColorHost(
                        *L.sol, *L.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha, parity
                    );
                }
            }
        }
    }

    // Jacobi-preconditioned Chebyshev smoother of degree `degree`: one full-cell
    // fused residual+increment kernel per degree (plain-stencil bandwidth, no
    // colour split, one ghost fill per degree). A fixed polynomial in the
    // symmetric operator -> symmetric linear smoother, CG-safe by construction.
    void chebyshevSmooth(std::size_t l, int degree) const
    {
        if (degree <= 0)
        {
            return;
        }
        const GmgLevelT<T>& L = levels_[l];
        const double b = L.lambdaMax;
        const double a = b / kChebEigRatio;
        const double theta = 0.5 * (b + a);
        const double delta = 0.5 * (b - a);
        const double sigma = theta / delta;
        double rho = 1.0 / sigma;
        for (int m = 0; m < degree; ++m)
        {
            fillGhosts(L, static_cast<int>(l));
            double ca = 0.0;
            double cb = 0.0;
            bool readOld = false;
            if (m == 0)
            {
                cb = 1.0 / theta; // d = (1/theta) D^{-1} r
            }
            else
            {
                const double rhoNew = 1.0 / (2.0 * sigma - rho);
                ca = rho * rhoNew; // d = ca * d + cb * D^{-1} r
                cb = 2.0 * rhoNew / delta;
                readOld = true;
                rho = rhoNew;
            }
            if (onDevice_)
            {
                prof::Timer t("gmg.cheb", static_cast<int>(l));
                gmgChebComputeDDevice(
                    *L.sol,
                    *L.rhs,
                    *L.ux,
                    *L.lx,
                    *L.uy,
                    *L.ly,
                    *L.uz,
                    *L.lz,
                    *L.alpha,
                    *L.chebD,
                    static_cast<T>(ca),
                    static_cast<T>(cb),
                    readOld
                );
            }
            else
            {
                gmgChebComputeDHost(
                    *L.sol,
                    *L.rhs,
                    *L.ux,
                    *L.lx,
                    *L.uy,
                    *L.ly,
                    *L.uz,
                    *L.lz,
                    *L.alpha,
                    *L.chebD,
                    static_cast<T>(ca),
                    static_cast<T>(cb),
                    readOld
                );
                amrex::Gpu::streamSynchronize();
            }
            GmgFab<T>::Saxpy(*L.sol, T(1), *L.chebD, 0, 0, 1, amrex::IntVect(0)); // sol += d
            if (!onDevice_)
            {
                amrex::Gpu::streamSynchronize();
            }
        }
    }

    // lambda_max(D^{-1}A) on level l via power iteration on a checkerboard seed
    // (near the top eigenvector). Returns the estimate inflated by kChebSafety
    // so the Chebyshev interval upper bound is not undershot.
    double estimateLambdaMax(std::size_t l) const
    {
        const GmgLevelT<T>& L = levels_[l];
        GmgFab<T>& v = *L.sol;   // scratch (1 ghost)
        GmgFab<T>& w = *L.chebD; // scratch (0 ghost)
        if (onDevice_)
        {
            gmgFillCheckerDevice(v);
        }
        else
        {
            gmgFillCheckerHost(v);
            amrex::Gpu::streamSynchronize();
        }
        double norm = gmgNorm2(v);
        v.mult(static_cast<T>(1.0 / norm), 0, 1, 0);
        double lambda = 0.0;
        for (int it = 0; it < kPowerIters; ++it)
        {
            fillGhosts(L, static_cast<int>(l));
            if (onDevice_)
            {
                gmgDinvApplyDevice(v, w, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha);
            }
            else
            {
                gmgDinvApplyHost(v, w, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha);
                amrex::Gpu::streamSynchronize();
            }
            lambda = gmgNorm2(w); // v is unit-norm -> ||D^{-1}A v|| ~ lambda_max
            if (lambda <= 0.0)
            {
                break;
            }
            if (onDevice_)
            {
                gmgConvertCopyDevice(v, w); // v <- w
            }
            else
            {
                gmgConvertCopyHost(v, w);
                amrex::Gpu::streamSynchronize();
            }
            v.mult(static_cast<T>(1.0 / lambda), 0, 1, 0);
        }
        v.setVal(T(0)); // leave sol clean for the V-cycle
        amrex::Gpu::streamSynchronize();
        return lambda * kChebSafety;
    }

    // One V-cycle correcting levels_[l].sol in place (warm start allowed, so
    // repeated cycles at l = 0 compose correctly).
    void vcycle(std::size_t l) const
    {
        const GmgLevelT<T>& L = levels_[l];
        if (l + 1 == levels_.size())
        {
            // Tiny grid: smoothing is cheap; forward + reversed halves keep
            // the coarsest "solve" self-adjoint (RB-GS; Chebyshev is symmetric
            // regardless, so the two halves just compose into a degree-2*n poly).
            smooth(l, coarsestSweeps_ / 2, false);
            smooth(l, coarsestSweeps_ / 2, true);
            return;
        }
        smooth(l, preSweeps_, false);
        fillGhosts(L, static_cast<int>(l));
        const GmgLevelT<T>& C = levels_[l + 1];
        // Fused residual + restriction: coarse rhs = avg(rhs - A sol) computed on
        // the fly, saving the separate fine-grid residual read+write (M4 item 3).
        {
            prof::Timer t("gmg.residrestrict", static_cast<int>(l));
            if (onDevice_)
            {
                gmgResidRestrictDevice(
                    *L.sol, *L.rhs, *C.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha
                );
            }
            else
            {
                gmgResidRestrictHost(
                    *L.sol, *L.rhs, *C.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha
                );
            }
            C.sol->setVal(0.0);
        }
        if (!onDevice_)
        {
            amrex::Gpu::streamSynchronize(); // setVal before host loops
        }
        vcycle(l + 1);
        {
            prof::Timer t("gmg.prolong", static_cast<int>(l));
            if (onDevice_)
            {
                gmgProlongAddDevice(*C.sol, *L.sol);
            }
            else
            {
                gmgProlongAddHost(*C.sol, *L.sol);
            }
        }
        smooth(l, postSweeps_, true);
    }

    BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool onDevice_ = false;
    int nCycles_ = 1;
    int preSweeps_ = 2;
    int postSweeps_ = 2;
    int coarsestSweeps_ = 8;
    bool useCheb_ = false;
    std::vector<GmgLevelT<T>> levels_;
};

} // namespace blockamr::solvers
