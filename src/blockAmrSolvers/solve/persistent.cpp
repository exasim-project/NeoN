// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "persistent.hpp"

#include <AMReX_Arena.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelContext.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../common/profiling.hpp"
#include "../common/transfer.hpp"
#include "../gmgKokkos/precond.hpp"
#include "../krylov/executor.hpp"
#include "../krylov/logging.hpp"
#include "../krylov/mixed_precision.hpp"
#include "../operators/csr.hpp"
#include "../operators/face_coeff_op.hpp"
#include "../operators/mlmg_ops.hpp"

namespace blockamr::solvers
{

// The one path whose flat Ginkgo vectors were never given the local sizing plus
// distributed view that the Krylov paths got (makeGlobalVec below): the CSR
// assembly, which is single-box only and so on >1 rank puts every row on one rank
// while the others hold none. It still sizes by the global cell count and reduces
// rank-locally, so it computes a wrong answer rather than a slow one. Refusing is
// the whole point -- silence here is what let the residual norm be rank-local for
// months.
static void requireSingleRank(const char* what)
{
    if (amrex::ParallelContext::NProcsSub() > 1)
    {
        throw std::runtime_error(
            std::string(what)
            + " is single-rank only; it has not been converted to the "
              "distributed vectors the fp64 Krylov path uses, and on more than one rank its "
              "reductions are rank-local. Run it on one rank, or use solver='cg'/'gmg'."
        );
    }
}

// Declared in dist_vec.hpp. The two bodies are spelled out rather than shared
// through a template helper because nvcc rejects ANY template signature that
// returns shared_ptr<LinOp> here (see the note in dist_vec.hpp); the
// duplication is the price of keeping this compilable in a CUDA build.
std::shared_ptr<gko::LinOp> makeGlobalVec(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type nGlobal,
    gko::matrix::Dense<double>* local
)
{
    // Aliases `local`: owns the Dense object, not the data.
    const auto nLocal = local->get_size()[0];
    auto view = gko::matrix::Dense<double>::create(
        exec, gko::dim<2> {nLocal, 1}, gko::make_array_view(exec, nLocal, local->get_values()), 1
    );
#if GINKGO_BUILD_MPI
    if (amrex::ParallelContext::NProcsSub() > 1)
    {
        return gko::share(DistVec<double>::create(
            exec,
            gko::experimental::mpi::communicator(amrex::ParallelContext::CommunicatorSub()),
            gko::dim<2> {nGlobal, 1},
            std::move(view)
        ));
    }
#else
    (void)nGlobal;
#endif
    return gko::share(std::move(view));
}

std::shared_ptr<gko::LinOp> makeGlobalVec(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type nGlobal,
    gko::matrix::Dense<float>* local
)
{
    // Aliases `local`: owns the Dense object, not the data.
    const auto nLocal = local->get_size()[0];
    auto view = gko::matrix::Dense<float>::create(
        exec, gko::dim<2> {nLocal, 1}, gko::make_array_view(exec, nLocal, local->get_values()), 1
    );
#if GINKGO_BUILD_MPI
    if (amrex::ParallelContext::NProcsSub() > 1)
    {
        return gko::share(DistVec<float>::create(
            exec,
            gko::experimental::mpi::communicator(amrex::ParallelContext::CommunicatorSub()),
            gko::dim<2> {nGlobal, 1},
            std::move(view)
        ));
    }
#else
    (void)nGlobal;
#endif
    return gko::share(std::move(view));
}

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

    if (bcOffset_)
    {
        // Inhomogeneous domain BCs make the boundary operator AFFINE,
        // L(x) = A x + c0. op_ is the linear part alone, so the system to solve
        // is A x = rhs - c0; subtracting the offset here is the whole fold, and
        // it also makes the residual reported below the residual of L (the
        // report uses this same b_). c0 was refreshed by the subclass just
        // before this call, so an in-place bc_data update takes effect.
        auto negOne = gko::initialize<Dense>({-1.0}, exec_);
        b_->add_scaled(negOne, bcOffset_);
    }

    if (projectNullspace_)
    {
        // Singular system with the constant nullspace (e.g. fully-periodic
        // pure Poisson): make the rhs consistent by removing its mean, and
        // keep the initial guess in the mean-zero subspace so CG stays there.
        subtractMean(bGlobal_.get());
        subtractMean(xGlobal_.get());
    }

    {
        prof::Timer t("solve.krylov");
        solver_->apply(bGlobal_, xGlobal_);
    }

    if (projectNullspace_)
    {
        // Pin the arbitrary constant: return the mean-zero representative
        // (also removes any roundoff drift out of the subspace).
        subtractMean(xGlobal_.get());
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
    // Cloned from the GLOBAL view, so the norm below reduces across ranks in
    // exactly the way the stopping criterion did.
    auto res = bGlobal_->clone();
    auto one = gko::initialize<Dense>({1.0}, exec_);
    auto negOne = gko::initialize<Dense>({-1.0}, exec_);
    op_->apply(negOne, xGlobal_, one, res);
    const double resNorm = (norm_ == NormKind::linf) ? normInf(res.get()) : globalNorm2(res.get());

    return makeResultDict(*logger_, *resLogger_, resNorm);
}

PersistentSolver::PersistentSolver(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    gko::size_type nLocal,
    bool allocDense
)
    : exec_(std::move(exec)), onDevice_(exec_->get_master().get() != exec_.get()), n_(n),
      nLocal_(nLocal)
{
    if (allocDense)
    {
        b_ = Dense::create(exec_, gko::dim<2> {nLocal_, 1});
        x_ = Dense::create(exec_, gko::dim<2> {nLocal_, 1});
        bGlobal_ = makeGlobalVec(exec_, n_, b_.get());
        xGlobal_ = makeGlobalVec(exec_, n_, x_.get());
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
        ones_ = Dense::create(exec_, gko::dim<2> {nLocal_, 1});
        ones_->fill(1.0);
        onesGlobal_ = makeGlobalVec(exec_, n_, ones_.get());
    }
}

void PersistentSolver::subtractMean(gko::LinOp* v)
{
    // n_ (global) is the right divisor for a sum that is now also global.
    const double sum = globalDot(v, onesGlobal_.get());
    auto negMean = gko::initialize<Dense>({-sum / static_cast<double>(n_)}, exec_);
    // add_scaled is elementwise, so it runs on the rank's own slice.
    localView<double>(v)->add_scaled(negMean, ones_);
}

FaceCoeffSolver::FaceCoeffSolver(
    const NeoN::Executor& executor,
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
    const std::string& gmg_coeff_precision,
    double gmg_omega,
    int gmg_agg_l0_size,
    bool symmetric,
    const std::string& gmg_bottom_solver,
    int gmg_bottom_max_iter,
    double gmg_bottom_rtol,
    double mp_inner_rtol,
    int mp_inner_max_iter,
    const std::string& norm,
    const amrex::MultiFab* bc_data
)
    : PersistentSolver(
        makeExecutor(executor),
        static_cast<gko::size_type>(alpha->boxArray().numPts()),
        localCount(*alpha),
        solver != "gmg"
    )
{
    // A separate coefficient precision exists in the Kokkos hierarchy alone. Named
    // rather than ignored: the shipped GmgPrecondT stores one type per level, so
    // accepting the option there would report a narrowed-coefficient timing for a
    // hierarchy that never narrowed anything.
    if (!gmg_coeff_precision.empty() && precond != "gmg_kokkos")
    {
        throw std::runtime_error(
            "FaceCoeffSolver: gmg_coeff_precision needs precond='gmg_kokkos' (the shipped GMG "
            "hierarchy stores its coefficients in the same type as its fields)"
        );
    }

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
    if (bc_data != nullptr)
    {
        checkBcData(*bc_data, *alpha, bcArr, "FaceCoeffSolver");
    }

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
            bcData_ = bc_data;
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
            if (bc_data != nullptr)
            {
                ownedBcData_ = pinnedCopy(*bc_data);
                bcData_ = ownedBcData_.get();
            }
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
            gmg_omega,
            symmetric,
            gmg_bottom_solver,
            gmg_bottom_max_iter,
            gmg_bottom_rtol
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
        bcArr,
        bc_data
    ));
    if (bc_data != nullptr)
    {
        // The typed hook solve() calls to refresh c0, plus the vector to hold
        // it. op_ (set by build() below) keeps the operator alive.
        bcOffsetOp_ = op.get();
        bcOffset_ = Dense::create(exec_, gko::dim<2> {nLocal_, 1});
    }

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
            gmg_omega,
            symmetric,
            gmg_bottom_solver,
            gmg_bottom_max_iter,
            gmg_bottom_rtol
        );
        build(op, solver, max_iter, rtol, atol, project_nullspace, std::move(inner), norm);
        return;
    }

    std::shared_ptr<const gko::LinOp> pc;
    // Set only by precond="gmg_kokkos"; solver="mpir" needs it and says so.
    std::shared_ptr<bench::KokkosGmgApply> vcycle;
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
            gmg_omega,
            symmetric,
            gmg_bottom_solver,
            gmg_bottom_max_iter,
            gmg_bottom_rtol
        );
    }
    else if (precond == "gmg_kokkos")
    {
        // The same V-cycle as precond="gmg", under the optimised Kokkos launchers
        // (gmgKokkos/apply.hpp). A separate object rather than a mode of GmgPrecondT:
        // that one is the shipped baseline and stays untouched, so both can run in
        // one process and be compared directly.
        if (precond_mlmg != nullptr)
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' cannot be combined with precond_mlmg"
            );
        }
        // Refused rather than ignored, for the same reason every other
        // capability gap on this path is: accepting a knob that does nothing
        // reports a Krylov bottom in the configuration and runs fixed sweeps.
        // The ported V-cycle lives behind the bench fence and has no Ginkgo, so
        // GmgBottomOp cannot reach it; closing this means porting the bottom
        // solve to that side, not relaxing the check.
        if (gmg_bottom_solver != "smoother")
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' has no Krylov bottom solve, so "
                "gmg_bottom_solver='"
                + gmg_bottom_solver
                + "' would silently run gmg_coarsest_sweeps sweeps. Use "
                  "precond='gmg' for a Krylov bottom."
            );
        }
        // The Kokkos V-cycle carries the same symmetry assumptions the shipped one
        // does (an over-relaxed red-black sweep, a self-adjoint cycle), and has no
        // path that would honour symmetric=False.
        if (!symmetric)
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' assumes a symmetric operator; "
                "symmetric=False needs precond='gmg'"
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
        // Straight through, unvalidated here: makeKokkosGmgApply parses it and
        // throws on an unknown spelling, so a typo cannot quietly run fp64. This
        // is the only precond that has a bf16 hierarchy.
        opts.precision = gmg_precision;
        // Likewise unvalidated here beyond the guard above: makeKokkosGmgApply
        // rejects an unknown spelling and a coefficient type wider than the fields.
        opts.coeffPrecision = gmg_coeff_precision;
        // The parsed spec straight through: the ported V-cycle carries the same
        // homogeneous Dirichlet/Neumann reflection as precond="gmg", built once per
        // level as a device plan rather than as a per-box AMReX launch.
        opts.bc = bcArr;
        opts.aggLevel0Size = gmg_agg_l0_size;
        // Held in a local as well: solver="mpir" wraps the SAME hierarchy in an fp32
        // LinOp, and building it twice would double the setup and the device memory
        // for two views of one V-cycle.
        vcycle = std::shared_ptr<bench::KokkosGmgApply>(
            bench::makeKokkosGmgApply(geom, *alpha, *ux, *lx, *uy, *ly, *uz, *lz, opts)
        );
        pc = gko::share(GmgKokkosPrecond::create(exec_, n_, vcycle));
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
    // Mixed-precision iterative refinement. The OUTER loop is Ginkgo's Ir over the
    // fp64 operator -- it forms r = b - A x and runs the stopping test in fp64, so
    // the answer and the tolerance are the fp64 solver's -- and the inner correction
    // is a preconditioned Cg<float>. Expressed through the existing "ir" path
    // because Ir::with_generated_solver is exactly the hook needed: what changes is
    // only WHICH LinOp plays the inner solver.
    std::string krylov = solver;
    if (solver == "mpir")
    {
        if (!vcycle)
        {
            throw std::runtime_error(
                "FaceCoeffSolver: solver='mpir' needs precond='gmg_kokkos' (it is the only "
                "preconditioner with an fp32 apply)"
            );
        }
        auto op32 = gko::share(FaceCoeffOp32::create(
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
        auto pc32 = gko::share(GmgKokkosPrecond32::create(exec_, n_, vcycle));
        // l2 rather than the caller's norm: this is an INNER tolerance, not a
        // convergence claim, and ResidualNormInf is an fp64 criterion.
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> innerCriteria {
            gko::stop::Iteration::build()
                .with_max_iters(static_cast<gko::size_type>(mp_inner_max_iter))
                .on(exec_),
            gko::stop::ResidualNorm<float>::build()
                .with_baseline(gko::stop::mode::rhs_norm)
                .with_reduction_factor(static_cast<float>(mp_inner_rtol))
                .on(exec_)
        };
        auto cg32 = gko::share(gko::solver::Cg<float>::build()
                                   .with_criteria(innerCriteria)
                                   .with_generated_preconditioner(pc32)
                                   .on(exec_)
                                   ->generate(op32));
        pc = gko::share(MixedPrecisionSolve::create(exec_, n_, cg32));
        krylov = "ir";
    }
    build(op, krylov, max_iter, rtol, atol, project_nullspace, std::move(pc), norm);
}

nb::dict FaceCoeffSolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    if (gmgStationary_)
    {
        return gmgSolve(rhs, sol);
    }
    if (bcOffsetOp_ != nullptr)
    {
        // c0 = L(0), refreshed every solve: the BC data is REFERENCED, not copied,
        // on the device path, so an in-place update has to take effect exactly as
        // an in-place coefficient update does. One extra operator apply per solve,
        // which is the whole price of inhomogeneous BCs on the Krylov path.
        //
        // x_ is the zero source rather than a dedicated vector: PersistentSolver::
        // solve overwrites it with the initial guess as its first act, so the fold
        // costs one n-vector (bcOffset_) instead of two.
        x_->fill(0.0);
        bcOffsetOp_->applyBcOffset(x_.get(), bcOffset_.get());
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
    double gmg_omega,
    bool symmetric,
    const std::string& gmg_bottom_solver,
    int gmg_bottom_max_iter,
    double gmg_bottom_rtol
)
{
    // bf16 is named separately from an outright typo: it exists, but only for
    // precond='gmg_kokkos'. The shipped GmgPrecondT hierarchy is fp64/fp32, and
    // instantiating it for a storage-only type would mean porting its Chebyshev
    // smoother and lambda-max power iteration too.
    if (gmg_precision == "bf16")
    {
        throw std::runtime_error("FaceCoeffSolver: gmg_precision='bf16' needs precond='gmg_kokkos' "
                                 "(the shipped GMG hierarchy is fp64/fp32 only)");
    }
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
            gmg_omega,
            symmetric,
            gmg_bottom_solver,
            gmg_bottom_max_iter,
            gmg_bottom_rtol
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
        const amrex::Real* dx = geom_.CellSize();
        if (onDevice_)
        {
            if (bcData_ != nullptr)
            {
                fillDomainBcGhostsInhomDevice(mf, *bcData_, geom_.Domain(), bcArr_, dx);
            }
            else
            {
                fillDomainBcGhostsDevice(mf, geom_.Domain(), bcArr_);
            }
        }
        else
        {
            if (bcData_ != nullptr)
            {
                fillDomainBcGhostsInhomHost(mf, *bcData_, geom_.Domain(), bcArr_, dx);
            }
            else
            {
                fillDomainBcGhostsHost(mf, geom_.Domain(), bcArr_);
            }
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

    nb::dict out = makeResultDict(static_cast<std::int64_t>(cycles), rnorm, converged, history);

    // Convergence diagnostic. A stationary V-cycle contracts the residual by a
    // roughly CONSTANT factor per cycle, so makeResultDict's `contraction` (the
    // geometric mean of that factor) is the one number that says whether the
    // cycle is working -- and it says it even on a run that converged, which a
    // pass/fail flag cannot. Without it a caller sees only "did not converge in
    // max_iter" and cannot tell a V-cycle that is grinding at 0.97/cycle from
    // one that diverged on cycle two. Only this path attaches a `diagnostic` to
    // it, because only here is the number a stable property of the method.
    //
    // Reported, not printed: this path already returns a dict the caller reads,
    // and a std::cerr warning inside a solve is both unmissable in a sweep and
    // unactionable in a script.
    //
    // The threshold is a "look here" signal, not a tolerance, but it has to sit
    // BELOW the cases worth looking at. Measured at N=16 on the constant-coefficient
    // periodic problem, smoothing bottom: 0.070 at 1 box, 0.155 at 8 boxes, 0.594
    // at 64 boxes -- and only the last fails to converge in 30 cycles. With a Krylov
    // bottom all three sit at 0.058-0.070, which is what "healthy" looks like here.
    // So the degraded case is 4x the healthy rate, and a threshold anywhere in
    // between separates them; one decade per 3 cycles is the round number in that
    // gap, and still allows 3x the cycles a healthy V-cycle needs before it says
    // anything. Crossing it means something structural: most often a bottom grid
    // the smoother cannot solve (see gmg_bottom_solver), otherwise anisotropy or a
    // coefficient jump the hierarchy does not represent.
    constexpr double slowRho = 0.464; // 10^(-1/3), i.e. one decade per 3 cycles
    const double rho = nb::cast<double>(out["contraction"]);
    const char* diagnostic = "";
    if (cycles > 0 && rho >= 1.0)
    {
        diagnostic = "V-cycle is not contracting (residual grew or stalled). Check the "
                     "bottom solve (gmg_bottom_solver), cell aspect ratio and coefficient "
                     "contrast.";
    }
    else if (cycles > 1 && rho > slowRho)
    {
        diagnostic = "V-cycle is contracting slowly (worse than one decade per 3 cycles). "
                     "The usual cause is a bottom grid too large for gmg_coarsest_sweeps -- "
                     "try gmg_bottom_solver='cg' (or 'bicgstab' when symmetric=False).";
    }
    out["diagnostic"] = diagnostic;
    return out;
}

FaceCoeffCsrSolver::FaceCoeffCsrSolver(
    const NeoN::Executor& executor,
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
    const std::string& /*gmg_coeff_precision*/,
    double /*gmg_omega*/,
    int /*gmg_agg_l0_size*/,
    bool /*symmetric*/,
    const std::string& /*gmg_bottom_solver*/,
    int /*gmg_bottom_max_iter*/,
    double /*gmg_bottom_rtol*/,
    double /*mp_inner_rtol*/,
    int /*mp_inner_max_iter*/,
    const std::string& norm,
    const amrex::MultiFab* bc_data
)
    : PersistentSolver(
        makeExecutor(executor),
        static_cast<gko::size_type>(alpha->boxArray().numPts()),
        localCount(*alpha)
    )
{
    // The assembly is single-box only (csr.cpp), which on >1 rank would mean one
    // rank holding every row while the others hold none.
    requireSingleRank("FaceCoeffCsrSolver");
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
    // Unreachable via a valid bc (bc_data needs a non-periodic side, which the
    // check above already refuses), but named rather than ignored so the message
    // stays the one the caller needs if that ever changes.
    if (bc_data != nullptr)
    {
        throw std::runtime_error(
            "FaceCoeffCsrSolver: periodic boundaries only — bc_data needs FaceCoeffSolver"
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
