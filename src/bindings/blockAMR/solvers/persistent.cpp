// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "persistent.hpp"

#include <AMReX_Arena.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "csr.hpp"
#include "face_coeff_op.hpp"
#include "gmg_kokkos_precond.hpp"
#include "mlmg_ops.hpp"
#include "profiling.hpp"
#include "transfer.hpp"

namespace blockamr::solvers
{

nb::dict PersistentSolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    resLogger_->clear(); // per-call history
    {
        prof::Timer t("solve.pack");
        if (onDevice_)
        {
            gather_device(rhs, b_->get_values(), 1.0);
            gather_device(sol, x_->get_values(), 1.0);
            amrex::Gpu::streamSynchronize();
        }
        else
        {
            gather(rhs, b_->get_values(), 1.0);
            gather(sol, x_->get_values(), 1.0);
        }
    }

    if (projectNullspace_)
    {
        // Singular system with the constant nullspace (e.g. fully-periodic
        // pure Poisson): make the rhs consistent by removing its mean, and
        // keep the initial guess in the mean-zero subspace so CG stays there.
        subtractMean(b_.get());
        subtractMean(x_.get());
    }

    {
        prof::Timer t("solve.krylov");
        solver_->apply(b_, x_);
    }

    if (projectNullspace_)
    {
        // Pin the arbitrary constant: return the mean-zero representative
        // (also removes any roundoff drift out of the subspace).
        subtractMean(x_.get());
    }

    {
        prof::Timer t("solve.unpack");
        if (onDevice_)
        {
            exec_->synchronize();
            scatter_device(x_->get_const_values(), sol);
            amrex::Gpu::streamSynchronize();
        }
        else
        {
            scatter(x_->get_const_values(), sol);
        }
    }

    // Final residual ||b - A x|| for reporting, in the norm the solve stopped on
    // (so a reported res_norm is always comparable with the rtol that produced it).
    prof::Timer tRep("solve.report");
    auto res = b_->clone();
    auto one = gko::initialize<Dense>({1.0}, exec_);
    auto negOne = gko::initialize<Dense>({-1.0}, exec_);
    op_->apply(negOne, x_, one, res);
    double resNorm;
    if (norm_ == NormKind::linf)
    {
        resNorm = normInf(res.get());
    }
    else
    {
        auto norm = Dense::create(exec_, gko::dim<2> {1, 1});
        res->compute_norm2(norm);
        auto normHost = gko::clone(exec_->get_master(), norm);
        resNorm = normHost->at(0, 0);
    }

    return makeResultDict(*logger_, *resLogger_, resNorm);
}

PersistentSolver::PersistentSolver(
    std::shared_ptr<const gko::Executor> exec, gko::size_type n, bool allocDense
)
    : exec_(std::move(exec)), onDevice_(exec_->get_master().get() != exec_.get()), n_(n)
{
    if (allocDense)
    {
        b_ = Dense::create(exec_, gko::dim<2> {n_, 1});
        x_ = Dense::create(exec_, gko::dim<2> {n_, 1});
    }
}

void PersistentSolver::build(
    std::shared_ptr<gko::LinOp> op,
    const std::string& solver,
    int max_iter,
    double rtol,
    double atol,
    bool project_nullspace,
    std::shared_ptr<const gko::LinOp> precond,
    const std::string& norm
)
{
    norm_ = parseNorm(norm);
    op_ = std::move(op);
    solver_ = buildKrylov(solver, exec_, op_, max_iter, rtol, atol, std::move(precond), norm);
    logger_ = gko::share(gko::log::Convergence<double>::create());
    solver_->add_logger(logger_);
    resLogger_ = std::make_shared<ResidualHistoryLogger>();
    solver_->add_logger(resLogger_);
    projectNullspace_ = project_nullspace;
    if (projectNullspace_)
    {
        ones_ = Dense::create(exec_, gko::dim<2> {n_, 1});
        ones_->fill(1.0);
    }
}

void PersistentSolver::subtractMean(Dense* v)
{
    auto sum = Dense::create(exec_, gko::dim<2> {1, 1});
    v->compute_dot(ones_, sum);
    auto sumHost = gko::clone(exec_->get_master(), sum);
    auto negMean = gko::initialize<Dense>({-sumHost->at(0, 0) / static_cast<double>(n_)}, exec_);
    v->add_scaled(negMean, ones_);
}

FaceCoeffSolver::FaceCoeffSolver(
    const std::string& executor,
    amrex::Geometry geom,
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    const std::string& solver,
    int max_iter,
    double rtol,
    double atol,
    bool project_nullspace,
    MLMG* precond_mlmg,
    int precond_cycles,
    const std::vector<std::string>& bc,
    const std::string& precond,
    int gmg_pre_sweeps,
    int gmg_post_sweeps,
    int gmg_coarsest_sweeps,
    int gmg_max_levels,
    int gmg_min_bottom,
    const std::string& gmg_smoother,
    const std::string& gmg_precision,
    double gmg_omega,
    const std::string& norm
)
    : PersistentSolver(
        makeExecutor(executor),
        static_cast<gko::size_type>(alpha->boxArray().numPts()),
        solver != "gmg"
    )
{
    // CG-safety: the V-cycle is a symmetric (SPD) preconditioner only when
    // the post-smoother is the adjoint of the pre-smoother, which requires
    // equal pre/post counts. With asymmetric counts CG's assumption breaks;
    // warn but allow (usable as a stationary/flexible-CG smoother). The native
    // stationary solver (solver="gmg") is NOT CG, so asymmetric sweeps there
    // are legitimate and never warn (this guard requires solver=="cg").
    if (precond == "gmg" && solver == "cg" && gmg_pre_sweeps != gmg_post_sweeps)
    {
        std::cerr << "FaceCoeffSolver: warning — gmg_pre_sweeps (" << gmg_pre_sweeps
                  << ") != gmg_post_sweeps (" << gmg_post_sweeps
                  << ") makes the V-cycle non-symmetric; CG may stall or diverge. "
                     "Use equal counts for a CG-safe preconditioner.\n";
    }
    const BcArray bcArr = parseBc(bc, geom, "FaceCoeffSolver");

    // solver="gmg": native stationary geometric-multigrid solver
    // (x <- x + V(b - A x) until tolerance). The GMG V-cycle IS the solver,
    // so `precond` is ignored; the hierarchy is built directly and the whole
    // iteration runs on AMReX fabs (see gmgSolve). No Ginkgo Krylov object.
    if (solver == "gmg")
    {
        if (precond_mlmg != nullptr)
        {
            throw std::runtime_error(
                "FaceCoeffSolver: solver='gmg' cannot be combined with precond_mlmg"
            );
        }
        gmgStationary_ = true;
        if (onDevice_)
        {
            // Device residual kernel reads the caller's device coefficients
            // directly (in-place updates are seen, like FaceCoeffOp).
            alpha_ = alpha;
            ux_ = ux;
            lx_ = lx;
            uy_ = uy;
            ly_ = ly;
            uz_ = uz;
            lz_ = lz;
        }
        else
        {
            // Host residual loops can't read device memory: stage the
            // coefficients to pinned once (solve-constant, cf. FaceCoeffOp).
            ownedCoeff_ = {
                pinnedCopy(*alpha),
                pinnedCopy(*ux),
                pinnedCopy(*lx),
                pinnedCopy(*uy),
                pinnedCopy(*ly),
                pinnedCopy(*uz),
                pinnedCopy(*lz)
            };
            alpha_ = ownedCoeff_[0].get();
            ux_ = ownedCoeff_[1].get();
            lx_ = ownedCoeff_[2].get();
            uy_ = ownedCoeff_[3].get();
            ly_ = ownedCoeff_[4].get();
            uz_ = ownedCoeff_[5].get();
            lz_ = ownedCoeff_[6].get();
        }
        geom_ = geom;
        bcArr_ = bcArr;
        hasPhysBc_ = std::any_of(bcArr.begin(), bcArr.end(), [](int b) { return b != 0; });
        maxIter_ = max_iter;
        rtol_ = rtol;
        atol_ = atol;
        // The stationary loop runs its own stopping test, so build() -- which is
        // where the Krylov path records the norm -- is never reached here.
        norm_ = parseNorm(norm);
        projectNull_ = project_nullspace;
        gmgOwner_ = buildGmgHierarchy(
            alpha,
            ux,
            lx,
            uy,
            ly,
            uz,
            lz,
            geom,
            bcArr,
            precond_cycles,
            gmg_pre_sweeps,
            gmg_post_sweeps,
            gmg_coarsest_sweeps,
            gmg_max_levels,
            gmg_min_bottom,
            gmg_smoother,
            gmg_precision,
            gmg_omega
        );
        const amrex::BoxArray& ba = alpha->boxArray();
        const amrex::DistributionMapping& dm = alpha->DistributionMap();
        if (onDevice_)
        {
            xWork_ = std::make_shared<amrex::MultiFab>(ba, dm, 1, 1);
        }
        else
        {
            xWork_ = std::make_shared<amrex::MultiFab>(
                ba, dm, 1, 1, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
            );
            rhsPinned_ = std::make_shared<amrex::MultiFab>(
                ba, dm, 1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
            );
        }
        return;
    }

    auto op = gko::share(FaceCoeffOp::create(
        exec_,
        alpha->boxArray(),
        alpha->DistributionMap(),
        geom,
        n_,
        alpha,
        ux,
        lx,
        uy,
        ly,
        uz,
        lz,
        bcArr
    ));

    // solver="ir": Ginkgo iterative refinement (gko::solver::Ir<double>) whose
    // system matrix is the FaceCoeffOp above and whose inner solver is the
    // generated GMG V-cycle LinOp (with_generated_solver, relaxation 1.0). Like
    // solver="gmg" it implies the GMG hierarchy and ignores `precond`; unlike it
    // the loop runs through Ginkgo (Dense pack/unpack + Convergence logger kept),
    // so the measured overhead across the LinOp boundaries vs the native gmg loop
    // is part of the deliverable — this variant does NOT fuse across it.
    if (solver == "ir")
    {
        if (precond_mlmg != nullptr)
        {
            throw std::runtime_error(
                "FaceCoeffSolver: solver='ir' cannot be combined with precond_mlmg"
            );
        }
        auto inner = buildGmgHierarchy(
            alpha,
            ux,
            lx,
            uy,
            ly,
            uz,
            lz,
            geom,
            bcArr,
            precond_cycles,
            gmg_pre_sweeps,
            gmg_post_sweeps,
            gmg_coarsest_sweeps,
            gmg_max_levels,
            gmg_min_bottom,
            gmg_smoother,
            gmg_precision,
            gmg_omega
        );
        build(op, solver, max_iter, rtol, atol, project_nullspace, std::move(inner), norm);
        return;
    }

    std::shared_ptr<const gko::LinOp> pc;
    if (precond == "gmg")
    {
        if (precond_mlmg != nullptr)
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg' cannot be combined with precond_mlmg"
            );
        }
        pc = buildGmgHierarchy(
            alpha,
            ux,
            lx,
            uy,
            ly,
            uz,
            lz,
            geom,
            bcArr,
            precond_cycles,
            gmg_pre_sweeps,
            gmg_post_sweeps,
            gmg_coarsest_sweeps,
            gmg_max_levels,
            gmg_min_bottom,
            gmg_smoother,
            gmg_precision,
            gmg_omega
        );
    }
    else if (precond == "gmg_kokkos")
    {
        // The same V-cycle as precond="gmg", under the optimised Kokkos launchers
        // (bench/gmg_apply.hpp). A separate object rather than a mode of GmgPrecondT:
        // that one is the shipped baseline and stays untouched, so both can run in
        // one process and be compared directly.
        if (precond_mlmg != nullptr)
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' cannot be combined with precond_mlmg"
            );
        }
        if (gmg_smoother != "rbgs")
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' has only the red-black smoother, not '"
                + gmg_smoother + "'"
            );
        }
        bench::KokkosGmgOpts opts;
        opts.cycles = precond_cycles;
        opts.preSweeps = gmg_pre_sweeps;
        opts.postSweeps = gmg_post_sweeps;
        opts.coarsestSweeps = gmg_coarsest_sweeps;
        opts.maxLevels = gmg_max_levels;
        opts.minBottom = gmg_min_bottom;
        opts.omega = gmg_omega;
        opts.fp32 = (gmg_precision == "fp32");
        // The parsed spec straight through: the ported V-cycle carries the same
        // homogeneous Dirichlet/Neumann reflection as precond="gmg", built once per
        // level as a device plan rather than as a per-box AMReX launch.
        opts.bc = bcArr;
        pc = gko::share(GmgKokkosPrecond::create(
            exec_,
            n_,
            std::shared_ptr<bench::KokkosGmgApply>(
                bench::makeKokkosGmgApply(geom, *alpha, *ux, *lx, *uy, *ly, *uz, *lz, opts)
            )
        ));
    }
    else if (precond == "mlmg" || precond == "none")
    {
        // precond_mlmg alone implies "mlmg" (pre-existing behaviour).
        if (precond == "mlmg" && precond_mlmg == nullptr)
        {
            throw std::runtime_error("FaceCoeffSolver: precond='mlmg' requires precond_mlmg");
        }
        if (precond_mlmg != nullptr)
        {
            pc = gko::share(MlmgPrecond::create(
                exec_, precond_mlmg, alpha->boxArray(), alpha->DistributionMap(), n_, precond_cycles
            ));
        }
    }
    else
    {
        throw std::runtime_error(
            "FaceCoeffSolver: unknown precond '" + precond
            + "' (expected 'none', 'mlmg', 'gmg' or 'gmg_kokkos')"
        );
    }
    build(op, solver, max_iter, rtol, atol, project_nullspace, std::move(pc), norm);
}

nb::dict FaceCoeffSolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    if (gmgStationary_)
    {
        return gmgSolve(rhs, sol);
    }
    return PersistentSolver::solve(rhs, sol);
}

std::shared_ptr<const gko::LinOp> FaceCoeffSolver::buildGmgHierarchy(
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    const amrex::Geometry& geom,
    const BcArray& bcArr,
    int precond_cycles,
    int gmg_pre_sweeps,
    int gmg_post_sweeps,
    int gmg_coarsest_sweeps,
    int gmg_max_levels,
    int gmg_min_bottom,
    const std::string& gmg_smoother,
    const std::string& gmg_precision,
    double gmg_omega
)
{
    if (gmg_precision != "fp64" && gmg_precision != "fp32")
    {
        throw std::runtime_error(
            "FaceCoeffSolver: unknown gmg_precision '" + gmg_precision
            + "' (expected 'fp64' or 'fp32')"
        );
    }
    auto makeGmg = [&](auto tag) -> std::shared_ptr<const gko::LinOp>
    {
        using T = decltype(tag);
        auto p = GmgPrecondT<T>::create(
            exec_,
            alpha->boxArray(),
            alpha->DistributionMap(),
            geom,
            n_,
            alpha,
            ux,
            lx,
            uy,
            ly,
            uz,
            lz,
            bcArr,
            precond_cycles,
            gmg_pre_sweeps,
            gmg_post_sweeps,
            gmg_coarsest_sweeps,
            gmg_max_levels,
            gmg_min_bottom,
            gmg_smoother,
            gmg_omega
        );
        gmgMf_ = p.get(); // GmgPrecondT<T>* -> const GmgApplyMf* (kept alive by the return)
        return gko::share(std::move(p));
    };
    return (gmg_precision == "fp32") ? makeGmg(float {}) : makeGmg(double {});
}

void FaceCoeffSolver::fillGmgGhosts(amrex::MultiFab& mf) const
{
    mf.FillBoundary(geom_.periodicity());
    if (!onDevice_)
    {
        amrex::Gpu::streamSynchronize();
    }
    if (hasPhysBc_)
    {
        if (onDevice_)
        {
            fillDomainBcGhostsDevice(mf, geom_.Domain(), bcArr_);
        }
        else
        {
            fillDomainBcGhostsHost(mf, geom_.Domain(), bcArr_);
        }
    }
}

void FaceCoeffSolver::subtractMeanMf(amrex::MultiFab& mf) const
{
    const double mean = mf.sum(0) / static_cast<double>(n_);
    mf.plus(-mean, 0, 1);
}

nb::dict FaceCoeffSolver::gmgSolve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    // Warm start: x0 = incoming sol (do NOT zero — persistent-solver contract).
    amrex::MultiFab::Copy(*xWork_, sol, 0, 0, 1, 0);

    // Host residual loops can't read the device rhs: stage it to pinned once
    // per solve (it is constant across the cycle loop). Device path reads rhs
    // directly.
    const amrex::MultiFab* rhsUse = &rhs;
    if (!onDevice_)
    {
        amrex::MultiFab::Copy(*rhsPinned_, rhs, 0, 0, 1, 0);
        amrex::Gpu::streamSynchronize();
        rhsUse = rhsPinned_.get();
    }

    // Same stopping test in either norm: ||r|| <= max(rtol*||b||, atol), with
    // both measured consistently (norm="linf" is MLMG's ||.||_inf, so a solve can
    // be held to exactly MLMG's criterion -- see stop_norm_inf.hpp).
    const bool useInf = (norm_ == NormKind::linf);
    const double bNorm = useInf ? rhs.norminf(0, 1, amrex::IntVect(0)) : rhs.norm2(0);
    const double stopTol = std::max(rtol_ * bNorm, atol_);
    const double rhsMean = projectNull_ ? rhs.sum(0) / static_cast<double>(n_) : 0.0;
    if (projectNull_)
    {
        subtractMeanMf(*xWork_);
    }

    std::vector<double> history;
    // M3: one fused kernel forms the FP64 residual r = rhs - A x - rhsMean,
    // casts it into the (fp32/fp64) L0 rhs, and reduces ||r|| in double — no
    // separate FP64 residual MultiFab, norm pass, or convert-scatter. The
    // nullspace shift (rhsMean) folds into the same kernel, so the projected
    // path takes the fused route too (it only adds subtractMeanMf on x).
    auto computeResid = [&]() -> double
    {
        prof::Timer t("gmg.solve.resid");
        fillGmgGhosts(*xWork_);
        const ResidNorms nr = gmgMf_->residScatterNorm(
            *xWork_, *rhsUse, *ux_, *lx_, *uy_, *ly_, *uz_, *lz_, *alpha_, rhsMean
        );
        const double rn = useInf ? nr.maxabs : std::sqrt(nr.sumsq);
        history.push_back(rn);
        return rn;
    };

    double rnorm = computeResid();
    bool converged = rnorm <= stopTol;
    int cycles = 0;
    while (!converged && cycles < maxIter_)
    {
        {
            prof::Timer t("gmg.solve.vcycle");
            gmgMf_->vcycleGather(*xWork_); // x += V(r); the residual is already in L0 rhs
        }
        if (projectNull_)
        {
            subtractMeanMf(*xWork_);
        }
        ++cycles;
        rnorm = computeResid();
        converged = rnorm <= stopTol;
    }

    amrex::MultiFab::Copy(sol, *xWork_, 0, 0, 1, 0);

    return makeResultDict(static_cast<std::int64_t>(cycles), rnorm, converged, history);
}

FaceCoeffCsrSolver::FaceCoeffCsrSolver(
    const std::string& executor,
    amrex::Geometry geom,
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    const std::string& solver,
    int max_iter,
    double rtol,
    double atol,
    bool project_nullspace,
    MLMG* precond_mlmg,
    int precond_cycles,
    const std::vector<std::string>& bc,
    const std::string& precond,
    int /*gmg_pre_sweeps*/,
    int /*gmg_post_sweeps*/,
    int /*gmg_coarsest_sweeps*/,
    int /*gmg_max_levels*/,
    int /*gmg_min_bottom*/,
    const std::string& /*gmg_smoother*/,
    const std::string& /*gmg_precision*/,
    double /*gmg_omega*/,
    const std::string& norm
)
    : PersistentSolver(
        makeExecutor(executor), static_cast<gko::size_type>(alpha->boxArray().numPts())
    )
{
    // The CSR assembly wraps neighbour indices around the domain
    // (periodic-only); parseBc also rejects a non-periodic geometry.
    const BcArray bcArr = parseBc(bc, geom, "FaceCoeffCsrSolver");
    if (std::any_of(bcArr.begin(), bcArr.end(), [](int b) { return b != 0; }))
    {
        throw std::runtime_error(
            "FaceCoeffCsrSolver: periodic boundaries only — use FaceCoeffSolver "
            "for dirichlet/neumann bc"
        );
    }
    if (precond == "gmg")
    {
        throw std::runtime_error(
            "FaceCoeffCsrSolver: precond='gmg' is matrix-free only — use FaceCoeffSolver"
        );
    }
    if (precond != "none" && precond != "mlmg")
    {
        throw std::runtime_error(
            "FaceCoeffCsrSolver: unknown precond '" + precond + "' (expected 'none' or 'mlmg')"
        );
    }
    if (precond == "mlmg" && precond_mlmg == nullptr)
    {
        throw std::runtime_error("FaceCoeffCsrSolver: precond='mlmg' requires precond_mlmg");
    }
    auto op = assembleFaceCoeffCsr(exec_, geom, *alpha, *ux, *lx, *uy, *ly, *uz, *lz);
    std::shared_ptr<const gko::LinOp> pc;
    if (precond_mlmg != nullptr)
    {
        pc = gko::share(MlmgPrecond::create(
            exec_, precond_mlmg, alpha->boxArray(), alpha->DistributionMap(), n_, precond_cycles
        ));
    }
    build(op, solver, max_iter, rtol, atol, project_nullspace, std::move(pc), norm);
}

} // namespace blockamr::solvers
