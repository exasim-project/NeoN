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

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/linOpBase.hpp"
#include "NeoN/blockAmr/core/profiling.hpp"
#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"
#include "NeoN/blockAmr/core/gkoTypes.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/gmgBottom.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp"

namespace blockamr::la
{

// One multigrid level: geometry, rediscretised coefficients, work fields (sol has 1 ghost).
template<class T>
struct GmgLevelT
{
    amrex::Geometry geom;
    std::shared_ptr<GmgFab<T>> alpha, ux, lx, uy, ly, uz, lz;
    std::shared_ptr<GmgFab<T>> sol, rhs;
    std::shared_ptr<GmgFab<T>> chebD; // Chebyshev increment (only when smoother="chebyshev")
    double lambdaMax = 0.0;           // estimate of lambda_max(D^{-1}A) on this level
};

// Abstract hook exposing a GMG V-cycle as operations on FP64 MultiFabs, so the native
// stationary solver can drive GmgPrecondT<T> without knowing T. No Ginkgo vector involved.
class GmgApplyMf
{
public:

    virtual ~GmgApplyMf() = default;

    // Fused r = rhs - A*sol - shift -> (cast to T) L0 rhs; L0 sol := 0. Returns BOTH FP64
    // norms of r, so the norm authority stays double. `sol`'s ghosts must already be filled.
    virtual ResidNorms residScatterNorm(
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

    // Run nCycles_ V-cycles on the L0 rhs set by residScatterNorm, then x += the correction.
    virtual void vcycleGather(amrex::MultiFab& x) const = 0;
};

// Native GMG V-cycle preconditioner on the face-coefficient operator: z = M^{-1} r via
// n_cycles V-cycles of RB-GS smoothing, volume-average restriction and piecewise-constant
// prolongation (laws: gmgKernels.hpp; smoother measurements: report/blockamr-gmg-notes.md).
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
        const std::string& smoother,
        double omega,
        bool symmetric,
        const std::string& bottom_solver,
        int bottom_max_iter,
        double bottom_rtol
    )
        : AmrexLinOpBase<GmgPrecondT<T>>(exec, gko::dim<2> {n, n}), bc_(bc),
          hasPhysBc_(std::any_of(bc.begin(), bc.end(), [](int b) { return b != 0; })),
          onDevice_(exec->get_master().get() != exec.get()), nCycles_(n_cycles),
          preSweeps_(pre_sweeps), postSweeps_(post_sweeps), coarsestSweeps_(coarsest_sweeps),
          useCheb_(smoother == "chebyshev"), omega_(omega), symmetric_(symmetric)
    {
        if (omega <= 0.0 || omega >= 2.0)
        {
            throw std::runtime_error(
                "GmgPrecond: gmg_omega must lie in (0, 2) for a convergent "
                "relaxation (got "
                + std::to_string(omega) + ")"
            );
        }
        if (smoother != "rbgs" && smoother != "chebyshev")
        {
            throw std::runtime_error(
                "GmgPrecond: unknown gmg_smoother '" + smoother
                + "' (expected 'rbgs' or 'chebyshev')"
            );
        }
        validateBottomSolver(bottom_solver, symmetric);
        // Both rest on symmetry, so they are refused rather than warned about
        // (report/blockamr-gmg-notes.md#smoother).
        if (!symmetric)
        {
            if (omega != 1.0)
            {
                throw std::runtime_error(
                    "GmgPrecond: gmg_omega != 1.0 assumes a symmetric operator, but "
                    "symmetric=False was set (got omega = "
                    + std::to_string(omega) + "). Set gmg_omega=1.0."
                );
            }
            if (useCheb_)
            {
                throw std::runtime_error(
                    "GmgPrecond: gmg_smoother='chebyshev' builds its polynomial on a real "
                    "eigenvalue interval and assumes a symmetric operator, but "
                    "symmetric=False was set. Use gmg_smoother='rbgs'."
                );
            }
        }
        // Finest level: the coefficients are COPIED into this preconditioner's own arena, so
        // later caller writes go unseen — a stale preconditioner only costs iterations.
        levels_.push_back(makeLevel(ba, dm, geom));
        copyCoeff(*levels_[0].alpha, *alpha);
        copyCoeff(*levels_[0].ux, *ux);
        copyCoeff(*levels_[0].lx, *lx);
        copyCoeff(*levels_[0].uy, *uy);
        copyCoeff(*levels_[0].ly, *ly);
        copyCoeff(*levels_[0].uz, *uz);
        copyCoeff(*levels_[0].lz, *lz);

        // Coarsen by 2 while every box dimension stays divisible and the coarse domain keeps
        // >= min_bottom cells. alpha via gmgRestrict, faces via gmgCoarsenFace(scale = 4) —
        // two DIFFERENT laws, see gmgKernels.hpp.
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
            gmgRestrict(*fl.alpha, *c.alpha, onDevice_);
            gmgCoarsenFace(*fl.ux, *c.ux, 0, 4.0, onDevice_);
            gmgCoarsenFace(*fl.lx, *c.lx, 0, 4.0, onDevice_);
            gmgCoarsenFace(*fl.uy, *c.uy, 1, 4.0, onDevice_);
            gmgCoarsenFace(*fl.ly, *c.ly, 1, 4.0, onDevice_);
            gmgCoarsenFace(*fl.uz, *c.uz, 2, 4.0, onDevice_);
            gmgCoarsenFace(*fl.lz, *c.lz, 2, 4.0, onDevice_);
        }
        amrex::Gpu::streamSynchronize();

        // Chebyshev setup: the increment field plus a lambda_max estimate, per level.
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

        // Bottom solver, built once on the coarsest level. Null for "smoother".
        if (bottom_solver != "smoother")
        {
            const GmgLevelT<T>& B = levels_.back();
            // GLOBAL size for the operator, LOCAL for the vectors gather/scatter fill.
            const auto nBottom = static_cast<gko::size_type>(B.alpha->boxArray().numPts());
            const auto nBottomLocal = static_cast<gko::size_type>(localCount(*B.alpha));
            bottomOp_ = gko::share(GmgBottomOp<T>::create(
                exec,
                nBottom,
                B.alpha->boxArray(),
                B.alpha->DistributionMap(),
                B.geom,
                B.alpha,
                B.ux,
                B.lx,
                B.uy,
                B.ly,
                B.uz,
                B.lz,
                bc_,
                onDevice_
            ));
            bottomSolver_ =
                makeBottomSolver<T>(bottom_solver, exec, bottomOp_, bottom_max_iter, bottom_rtol);
            bottomB_ = gko::matrix::Dense<T>::create(exec, gko::dim<2> {nBottomLocal, 1});
            bottomX_ = gko::matrix::Dense<T>::create(exec, gko::dim<2> {nBottomLocal, 1});
            // The bottom Krylov's dots and norms need the outer solve's distributed views.
            bottomBGlobal_ = makeGlobalVec(exec, nBottom, bottomB_.get());
            bottomXGlobal_ = makeGlobalVec(exec, nBottom, bottomX_.get());
        }
    }

    // One kernel forms the FP64 residual, casts it into the T-typed L0 rhs and reduces its norm.
    ResidNorms residScatterNorm(
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
        ResidNorms norms = faceCoeffResidScatterNorm<T>(
            sol, rhs, ux, lx, uy, ly, uz, lz, alpha, shift, *L0.rhs, onDevice_
        );
        L0.sol->setVal(T(0)); // z0 = 0: apply M^{-1}, not a warm-started solve
        if (!onDevice_)
        {
            amrex::Gpu::streamSynchronize();
        }
        return norms;
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
                gmgConvertAdd(x, *L0.sol, onDevice_); // x += (double) L0 correction
                amrex::Gpu::streamSynchronize();
            }
        }
        else
        {
            for (int c = 0; c < nCycles_; ++c)
            {
                vcycle(0);
            }
            gmgConvertAdd(x, *L0.sol, onDevice_);
            amrex::Gpu::streamSynchronize();
        }
    }

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) visible here.
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
                scatter_device(localValues<double>(b), *L0.rhs);
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
                gather_device(*L0.sol, localValues<double>(x), 1.0);
                amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
            }
        }
        else
        {
            auto host = exec->get_master();
            auto bHost = gko::clone(host, localView<double>(b));
            scatter(bHost->get_const_values(), *L0.rhs);
            L0.sol->setVal(0.0);
            amrex::Gpu::streamSynchronize(); // setVal may run on the GPU stream
            for (int c = 0; c < nCycles_; ++c)
            {
                vcycle(0);
            }
            auto xHost = Dense::create(host, gko::dim<2> {localRows(x), 1});
            gather(*L0.sol, xHost->get_values(), 1.0);
            localView<double>(x)->copy_from(xHost);
        }
    }

private:

    // Chebyshev smooths [lambdaMax / kChebEigRatio, lambdaMax] and leaves the lower modes to
    // the coarse grid; a sweep put the minimum at 6 (report/blockamr-gmg-notes.md#smoother).
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

    // Copy the caller's FP64 coefficients into a level fab as T. On the reference path the
    // source may be device memory, hence the pinned staging copy.
    void copyCoeff(GmgFab<T>& dst, const amrex::MultiFab& src) const
    {
        if (onDevice_)
        {
            gmgConvertCopy(dst, src, onDevice_);
        }
        else
        {
            auto tmp = pinnedCopy(src);
            amrex::Gpu::streamSynchronize();
            gmgConvertCopy(dst, *tmp, onDevice_);
        }
    }

    // Fill sol's ghosts: FillBoundary, then the homogeneous reflection on domain faces. One bc
    // spec serves every level, the coarse domains being the fine one coarsened.
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

    // `reversed` only means something for RB-GS; `sweeps` is the sweep count or Cheb degree.
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

    // RB-GS sweeps; `reversed` (black-red) is the forward sweep's adjoint, which is what keeps
    // the V-cycle symmetric and so CG-safe.
    void rbgsSmooth(std::size_t l, int sweeps, bool reversed) const
    {
        const GmgLevelT<T>& L = levels_[l];
        const FaceCoeffs<T> fc {
            L.alpha.get(), L.ux.get(), L.lx.get(), L.uy.get(), L.ly.get(), L.uz.get(), L.lz.get()
        };
        for (int s = 0; s < sweeps; ++s)
        {
            for (int c = 0; c < 2; ++c)
            {
                const int parity = (reversed ? 1 + c : c) & 1;
                fillGhosts(L, static_cast<int>(l)); // the other colour changed — refresh ghosts
                if (onDevice_)
                {
                    prof::Timer t("gmg.gs", static_cast<int>(l));
                    gmgGsColor(*L.sol, *L.rhs, fc, parity, omega_, onDevice_);
                }
                else
                {
                    gmgGsColor(*L.sol, *L.rhs, fc, parity, omega_, onDevice_);
                }
            }
        }
    }

    // Jacobi-preconditioned Chebyshev: one fused kernel and one ghost fill per degree; a fixed
    // polynomial in a symmetric operator, so CG-safe by construction.
    void chebyshevSmooth(std::size_t l, int degree) const
    {
        if (degree <= 0)
        {
            return;
        }
        const GmgLevelT<T>& L = levels_[l];
        const FaceCoeffs<T> fc {
            L.alpha.get(), L.ux.get(), L.lx.get(), L.uy.get(), L.ly.get(), L.uz.get(), L.lz.get()
        };
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
                gmgChebComputeD(
                    *L.sol,
                    *L.rhs,
                    fc,
                    *L.chebD,
                    static_cast<T>(ca),
                    static_cast<T>(cb),
                    readOld,
                    onDevice_
                );
            }
            else
            {
                gmgChebComputeD(
                    *L.sol,
                    *L.rhs,
                    fc,
                    *L.chebD,
                    static_cast<T>(ca),
                    static_cast<T>(cb),
                    readOld,
                    onDevice_
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

    // lambda_max(D^{-1}A) by power iteration from a checkerboard seed, inflated by kChebSafety.
    double estimateLambdaMax(std::size_t l) const
    {
        const GmgLevelT<T>& L = levels_[l];
        GmgFab<T>& v = *L.sol;   // scratch (1 ghost)
        GmgFab<T>& w = *L.chebD; // scratch (0 ghost)
        const FaceCoeffs<T> fc {
            L.alpha.get(), L.ux.get(), L.lx.get(), L.uy.get(), L.ly.get(), L.uz.get(), L.lz.get()
        };
        gmgFillChecker(v, onDevice_);
        if (!onDevice_)
        {
            amrex::Gpu::streamSynchronize();
        }
        double norm = gmgNorm2(v);
        v.mult(static_cast<T>(1.0 / norm), 0, 1, 0);
        double lambda = 0.0;
        for (int it = 0; it < kPowerIters; ++it)
        {
            fillGhosts(L, static_cast<int>(l));
            gmgDinvApply(v, w, fc, onDevice_);
            if (!onDevice_)
            {
                amrex::Gpu::streamSynchronize();
            }
            lambda = gmgNorm2(w); // v is unit-norm -> ||D^{-1}A v|| ~ lambda_max
            if (lambda <= 0.0)
            {
                break;
            }
            gmgConvertCopy(v, w, onDevice_); // v <- w
            if (!onDevice_)
            {
                amrex::Gpu::streamSynchronize();
            }
            v.mult(static_cast<T>(1.0 / lambda), 0, 1, 0);
        }
        v.setVal(T(0)); // leave sol clean for the V-cycle
        amrex::Gpu::streamSynchronize();
        return lambda * kChebSafety;
    }

    // Krylov solve of the coarsest system A z = rhs, replacing the fixed sweeps; x starts at
    // zero, which keeps the bottom count reproducible (report/blockamr-gmg-notes.md#bottom).
    void bottomSolve(const GmgLevelT<T>& L) const
    {
        prof::Timer t("gmg.bottom");
        if (onDevice_)
        {
            gather_device(*L.rhs, bottomB_->get_values(), 1.0);
            amrex::Gpu::streamSynchronize(); // b read by Ginkgo next
        }
        else
        {
            gather(*L.rhs, bottomB_->get_values(), 1.0);
        }
        bottomX_->fill(T(0));
        bottomSolver_->apply(bottomBGlobal_, bottomXGlobal_);
        if (onDevice_)
        {
            this->get_executor()->synchronize(); // x written by Ginkgo
            scatter_device(bottomX_->get_const_values(), *L.sol);
        }
        else
        {
            scatter(bottomX_->get_const_values(), *L.sol);
        }
    }

    // One V-cycle correcting levels_[l].sol in place; a warm start is allowed.
    void vcycle(std::size_t l) const
    {
        const GmgLevelT<T>& L = levels_[l];
        if (l + 1 == levels_.size())
        {
            if (bottomSolver_)
            {
                bottomSolve(L);
                return;
            }
            // Forward + reversed halves keep the coarsest "solve" self-adjoint.
            smooth(l, coarsestSweeps_ / 2, false);
            smooth(l, coarsestSweeps_ / 2, true);
            return;
        }
        smooth(l, preSweeps_, false);
        fillGhosts(L, static_cast<int>(l));
        const GmgLevelT<T>& C = levels_[l + 1];
        // Fused residual + restriction, saving the fine-grid residual read+write.
        {
            prof::Timer t("gmg.residrestrict", static_cast<int>(l));
            const FaceCoeffs<T> fc {
                L.alpha.get(),
                L.ux.get(),
                L.lx.get(),
                L.uy.get(),
                L.ly.get(),
                L.uz.get(),
                L.lz.get()
            };
            gmgResidRestrict(*L.sol, *L.rhs, *C.rhs, fc, onDevice_);
            C.sol->setVal(0.0);
        }
        if (!onDevice_)
        {
            amrex::Gpu::streamSynchronize(); // setVal before host loops
        }
        vcycle(l + 1);
        {
            prof::Timer t("gmg.prolong", static_cast<int>(l));
            gmgProlongAdd(*C.sol, *L.sol, onDevice_);
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
    // RB-SOR relaxation factor; 1.0 = plain Gauss-Seidel. Unused when useCheb_.
    double omega_ = 1.0;
    // Declared by the caller, never sniffed (report/blockamr-gmg-notes.md#smoother).
    bool symmetric_ = true;
    std::vector<GmgLevelT<T>> levels_;
    // Null when gmg_bottom_solver == "smoother" (the default): fixed sweeps.
    std::shared_ptr<const GmgBottomOp<T>> bottomOp_;
    std::shared_ptr<const gko::LinOp> bottomSolver_;
    mutable std::shared_ptr<gko::matrix::Dense<T>> bottomB_, bottomX_;
    std::shared_ptr<gko::LinOp> bottomBGlobal_, bottomXGlobal_;
};

} // namespace blockamr::la
