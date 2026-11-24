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
#include "NeoN/blockAmr/linearAlgebra/faceCoeffLevel.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/gmgBottom.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp"
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"

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

    // Fused r = in.rhs - A*in.sol - in.shift -> (cast to T) L0 rhs; L0 sol := 0. Returns BOTH
    // FP64 norms of r, so the norm authority stays double. `in.sol`'s ghosts must already be
    // filled. The system arrives as the GmgResidualInput the kernel reads (gmgKernels.hpp), so
    // the nine loose MultiFabs -- six of them the transposable ux/lx/uy/ly/uz/lz -- are named
    // once, at the caller, with designated initialisers.
    virtual ResidNorms residScatterNorm(const GmgResidualInput& in) const = 0;

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
        gko::size_type n,
        const FaceCoeffLevel& level,
        BcArray bc,
        const GmgPrecondSpec& spec
    )
        : AmrexLinOpBase<GmgPrecondT<T>>(exec, gko::dim<2> {n, n}), bc_(bc),
          hasPhysBc_(std::any_of(bc.begin(), bc.end(), [](int b) { return b != 0; })),
          onDevice_(exec->get_master().get() != exec.get()), nCycles_(spec.nCycles),
          preSweeps_(spec.gmg.preSweeps), postSweeps_(spec.gmg.postSweeps),
          coarsestSweeps_(spec.gmg.coarsestSweeps), useCheb_(spec.gmg.smoother == "chebyshev"),
          omega_(spec.gmg.omega), symmetric_(spec.gmg.symmetric)
    {
        const GmgConfig& gmg = spec.gmg;
        validateOptions(gmg);
        // Finest level: the coefficients are COPIED into this preconditioner's own arena, so
        // later caller writes go unseen — a stale preconditioner only costs iterations.
        levels_.push_back(makeLevel(level.mesh.ba, level.mesh.dm, level.mesh.geom));
        copyCoeff(*levels_[0].alpha, *level.alpha);
        copyCoeff(*levels_[0].ux, level.upper[0]);
        copyCoeff(*levels_[0].lx, level.lower[0]);
        copyCoeff(*levels_[0].uy, level.upper[1]);
        copyCoeff(*levels_[0].ly, level.lower[1]);
        copyCoeff(*levels_[0].uz, level.upper[2]);
        copyCoeff(*levels_[0].lz, level.lower[2]);
        buildCoarseLevels(level.mesh.dm, gmg.maxLevels, gmg.minBottom);
        if (useCheb_)
        {
            setupChebyshev();
        }
        if (gmg.bottomSolver != "smoother")
        {
            setupBottomSolver(exec, gmg.bottomSolver, gmg.bottomMaxIter, gmg.bottomRtol);
        }
    }

    // One kernel forms the FP64 residual, casts it into the T-typed L0 rhs and reduces its norm.
    ResidNorms residScatterNorm(const GmgResidualInput& in) const override
    {
        const GmgLevelT<T>& L0 = levels_.front();
        ResidNorms norms = faceCoeffResidScatterNorm<T>(in, *L0.rhs, onDevice_);
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

    // Refuses, rather than warns about, an option combination the cycle cannot honour
    // (report/blockamr-gmg-notes.md#smoother).
    void validateOptions(const GmgConfig& gmg) const
    {
        const std::string& smoother = gmg.smoother;
        const double omega = gmg.omega;
        const bool symmetric = gmg.symmetric;
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
        validateBottomSolver(gmg.bottomSolver, symmetric);
        if (symmetric)
        {
            return;
        }
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

    // Coarsen by 2 while every box dimension stays divisible and the coarse domain keeps
    // >= min_bottom cells. alpha via gmgRestrict, faces via gmgCoarsenFace(scale = 4) —
    // two DIFFERENT laws, see gmgKernels.hpp.
    void buildCoarseLevels(const amrex::DistributionMapping& dm, int max_levels, int min_bottom)
    {
        while (canCoarsen(max_levels, min_bottom))
        {
            const GmgLevelT<T>& f = levels_.back();
            amrex::BoxArray cba = f.alpha->boxArray();
            cba.coarsen(2);
            const amrex::Geometry cgeom(
                amrex::coarsen(f.geom.Domain(), 2),
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
    }

    // Whether one more level fits below the current coarsest one.
    bool canCoarsen(int max_levels, int min_bottom) const
    {
        if (max_levels > 0 && static_cast<int>(levels_.size()) >= max_levels)
        {
            return false;
        }
        const GmgLevelT<T>& f = levels_.back();
        if (!f.alpha->boxArray().coarsenable(2, 2))
        {
            return false;
        }
        return amrex::coarsen(f.geom.Domain(), 2).shortside() >= min_bottom;
    }

    // The Chebyshev increment field plus a lambda_max estimate, per level.
    void setupChebyshev()
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

    // The Krylov bottom, built once on the coarsest level; left null for "smoother".
    void setupBottomSolver(
        std::shared_ptr<const gko::Executor> exec,
        const std::string& bottom_solver,
        int bottom_max_iter,
        double bottom_rtol
    )
    {
        const GmgLevelT<T>& B = levels_.back();
        // GLOBAL size for the operator, LOCAL for the vectors gather/scatter fill.
        const gko::size_type nBottom = gmgLevelRows(*B.alpha);
        const auto nBottomLocal = static_cast<gko::size_type>(localCount(*B.alpha));
        bottomOp_ = gko::share(GmgBottomOp<T>::create(
            exec,
            B.geom,
            OwnedFaceCoeffs<T> {B.alpha, B.ux, B.lx, B.uy, B.ly, B.uz, B.lz},
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

    // The level's 7 rediscretised coefficients as the view every kernel takes.
    static FaceCoeffs<T> levelCoeffs(const GmgLevelT<T>& L)
    {
        return {
            L.alpha.get(), L.ux.get(), L.lx.get(), L.uy.get(), L.ly.get(), L.uz.get(), L.lz.get()
        };
    }

    // RB-GS sweeps; `reversed` (black-red) is the forward sweep's adjoint, which is what keeps
    // the V-cycle symmetric and so CG-safe.
    void rbgsSmooth(std::size_t l, int sweeps, bool reversed) const
    {
        const GmgLevelT<T>& L = levels_[l];
        const FaceCoeffs<T> fc = levelCoeffs(L);
        for (int s = 0; s < sweeps; ++s)
        {
            for (int c = 0; c < 2; ++c)
            {
                const GsSweep sweep {(reversed ? 1 + c : c) & 1, omega_};
                fillGhosts(L, static_cast<int>(l)); // the other colour changed — refresh ghosts
                if (onDevice_)
                {
                    prof::Timer t("gmg.gs", static_cast<int>(l));
                    gmgGsColor(*L.sol, *L.rhs, fc, sweep, onDevice_);
                }
                else
                {
                    gmgGsColor(*L.sol, *L.rhs, fc, sweep, onDevice_);
                }
            }
        }
    }

    // The constants of one level's Chebyshev recurrence; `rho` advances with each degree.
    struct ChebRecurrence
    {
        double theta;
        double delta;
        double sigma;
        double rho;
    };

    // The recurrence over [lambdaMax / kChebEigRatio, lambdaMax], the band the smoother owns.
    static ChebRecurrence chebRecurrence(double lambdaMax)
    {
        const double b = lambdaMax;
        const double a = b / kChebEigRatio;
        const double theta = 0.5 * (b + a);
        const double delta = 0.5 * (b - a);
        const double sigma = theta / delta;
        return {theta, delta, sigma, 1.0 / sigma};
    }

    // Degree m's coefficients, advancing `rec`; degree 0 has no previous increment to fold in.
    static ChebStep<T> nextChebStep(int m, ChebRecurrence& rec)
    {
        if (m == 0)
        {
            return {T(0), static_cast<T>(1.0 / rec.theta), false}; // d = (1/theta) D^{-1} r
        }
        const double rhoNew = 1.0 / (2.0 * rec.sigma - rec.rho);
        const double ca = rec.rho * rhoNew; // d = ca * d + cb * D^{-1} r
        const double cb = 2.0 * rhoNew / rec.delta;
        rec.rho = rhoNew;
        return {static_cast<T>(ca), static_cast<T>(cb), true};
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
        const GmgSystem<T> sys {L.sol.get(), L.rhs.get(), levelCoeffs(L)};
        ChebRecurrence rec = chebRecurrence(L.lambdaMax);
        for (int m = 0; m < degree; ++m)
        {
            fillGhosts(L, static_cast<int>(l));
            const ChebStep<T> step = nextChebStep(m, rec);
            if (onDevice_)
            {
                prof::Timer t("gmg.cheb", static_cast<int>(l));
                gmgChebComputeD(sys, *L.chebD, step, onDevice_);
            }
            else
            {
                gmgChebComputeD(sys, *L.chebD, step, onDevice_);
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
        const FaceCoeffs<T> fc = levelCoeffs(L);
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
            gmgResidRestrict(*L.sol, *L.rhs, *C.rhs, levelCoeffs(L), onDevice_);
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
