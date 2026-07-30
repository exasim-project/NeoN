// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/linearAlgebra/solve/persistent.hpp"

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

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/profiling.hpp"
#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/gmgPrecond.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/precond.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/krylov.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/krylovSolver.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/logging.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/mixedPrecision.hpp"
#include "NeoN/blockAmr/linearAlgebra/precond.hpp"
#include "NeoN/blockAmr/linearAlgebra/sparse/csr.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/mlmgOps.hpp"

namespace blockamr::la
{

// REFUSED on >1 rank: the CSR path's flat vectors never got the local sizing plus distributed
// view makeGlobalVec gives the Krylov ones, so they size by the global cell count and reduce
// rank-locally -- a wrong answer rather than a slow one.
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

// Declared in distVec.hpp. Duplicated rather than templated because nvcc rejects any template
// signature returning shared_ptr<LinOp> here -- see the note in distVec.hpp.
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

SolveResult KrylovSolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
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
        // Inhomogeneous domain BCs make the boundary operator AFFINE, L(x) = A x + c0; op_ is
        // the linear part alone, so this fold solves A x = rhs - c0 and makes the residual
        // reported below the residual of L. c0 was refreshed by the subclass just above.
        auto negOne = gko::initialize<Dense>({-1.0}, exec_);
        b_->add_scaled(negOne, bcOffset_);
    }

    if (projectNullspace_)
    {
        // Constant-nullspace singular system: make the rhs consistent by removing its mean,
        // and keep the initial guess in the mean-zero subspace so CG stays there.
        subtractMean(bGlobal_.get());
        subtractMean(xGlobal_.get());
    }

    {
        prof::Timer t("solve.krylov");
        solver_->apply(bGlobal_, xGlobal_);
    }

    if (projectNullspace_)
    {
        // Pin the arbitrary constant: return the mean-zero representative.
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

    // Final residual ||b - A x|| in the norm the solve stopped on, so res_norm is comparable
    // with the rtol that produced it.
    prof::Timer tRep("solve.report");
    // From the GLOBAL view, so the norm reduces across ranks as the stopping criterion did.
    auto res = bGlobal_->clone();
    auto one = gko::initialize<Dense>({1.0}, exec_);
    auto negOne = gko::initialize<Dense>({-1.0}, exec_);
    op_->apply(negOne, xGlobal_, one, res);
    const double resNorm = (norm_ == NormKind::linf) ? normInf(res.get()) : globalNorm2(res.get());

    return makeSolveResult(*logger_, *resLogger_, resNorm);
}

KrylovSolver::KrylovSolver(
    std::shared_ptr<const gko::Executor> exec, gko::size_type n, gko::size_type nLocal
)
    : exec_(std::move(exec)), onDevice_(exec_->get_master().get() != exec_.get()), n_(n),
      nLocal_(nLocal)
{
    b_ = Dense::create(exec_, gko::dim<2> {nLocal_, 1});
    x_ = Dense::create(exec_, gko::dim<2> {nLocal_, 1});
    bGlobal_ = makeGlobalVec(exec_, n_, b_.get());
    xGlobal_ = makeGlobalVec(exec_, n_, x_.get());
}

void KrylovSolver::build(
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

void KrylovSolver::subtractMean(gko::LinOp* v)
{
    // n_ (global) is the right divisor for a sum that is now also global.
    const double sum = globalDot(v, onesGlobal_.get());
    auto negMean = gko::initialize<Dense>({-sum / static_cast<double>(n_)}, exec_);
    // add_scaled is elementwise, so it runs on the rank's own slice.
    localView<double>(v)->add_scaled(negMean, ones_);
}

namespace
{

// REFUSED, in all four modes that build a GMG hierarchy: a V-cycle already IS the
// preconditioner/solver, so an externally-built precond_mlmg has nothing to do. `what` names
// which combination fired.
void forbidPrecondMlmg(const SolverConfig& config, bool active, const char* what)
{
    if (active && config.precondMlmg != nullptr)
    {
        throw std::runtime_error(
            std::string("FaceCoeffSolver: ") + what + " cannot be combined with precond_mlmg"
        );
    }
}

} // namespace

// buildGmgHierarchy, shared by the three native-GMG call sites below, lives in
// linearAlgebra/precond.cpp so la::Matrix can build the same hierarchy (precond.hpp).

namespace
{

// Native stationary geometric-multigrid solver (solver="gmg"): x <- x + V(b - A x) run to
// tolerance, the whole loop on AMReX fabs with no Ginkgo Krylov object or Dense vector --
// which is why it does not derive from KrylovSolver.
class GmgStationarySolver : public ISolver
{
public:

    // Non-const because nonOwning() builds the grouped handles from a mutable field; the
    // members below stay const*.
    GmgStationarySolver(
        std::shared_ptr<const gko::Executor> exec,
        const NeoN::Executor& nexec,
        amrex::MultiFab* alpha,
        amrex::MultiFab* ux,
        amrex::MultiFab* lx,
        amrex::MultiFab* uy,
        amrex::MultiFab* ly,
        amrex::MultiFab* uz,
        amrex::MultiFab* lz,
        amrex::Geometry geom,
        const BcArray& bcArr,
        const SolverConfig& config
    );

    SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) override;

private:

    // Fill xWork_'s ghost layer exactly as FaceCoeffOp does, so the residual uses the same A.
    // With bcData_ the reflection is the INHOMOGENEOUS one, making the outer residual
    // rhs - L(x); the V-cycle below then solves A delta = rhs - L(x) with homogeneous fills.
    void fillGmgGhosts(amrex::MultiFab& mf) const;

    // mf -= mean(mf) over the valid region: the constant-nullspace projection.
    void subtractMeanMf(amrex::MultiFab& mf) const;

    std::shared_ptr<const gko::Executor> exec_;
    // Carried only for fillGmgGhosts' inhomogeneous fill; the V-cycle's own kernels take the
    // HostDeviceParallelFor path and cannot use it.
    NeoN::Executor nexec_ {NeoN::SerialExecutor {}};
    bool onDevice_;
    gko::size_type n_;
    const amrex::MultiFab* alpha_ = nullptr;
    const amrex::MultiFab* ux_ = nullptr;
    const amrex::MultiFab* lx_ = nullptr;
    const amrex::MultiFab* uy_ = nullptr;
    const amrex::MultiFab* ly_ = nullptr;
    const amrex::MultiFab* uz_ = nullptr;
    const amrex::MultiFab* lz_ = nullptr;
    amrex::Geometry geom_ {};
    BcArray bcArr_ {};
    bool hasPhysBc_ = false;
    const amrex::MultiFab* bcData_ = nullptr;
    std::shared_ptr<amrex::MultiFab> ownedBcData_;
    int maxIter_ = 0;
    double rtol_ = 0.0;
    double atol_ = 0.0;
    bool projectNull_ = false;
    NormKind norm_ = NormKind::l2;
    std::shared_ptr<const gko::LinOp> gmgOwner_;               // keeps the V-cycle hierarchy alive
    const GmgApplyMf* gmgMf_ = nullptr;                        // typed V-cycle hook into gmgOwner_
    std::shared_ptr<amrex::MultiFab> xWork_;                   // FP64 iterate (1 ghost)
    std::shared_ptr<amrex::MultiFab> rhsPinned_;               // pinned rhs stage (reference path)
    std::vector<std::shared_ptr<amrex::MultiFab>> ownedCoeff_; // pinned coeffs (reference path)
};

GmgStationarySolver::GmgStationarySolver(
    std::shared_ptr<const gko::Executor> exec,
    const NeoN::Executor& nexec,
    amrex::MultiFab* alpha,
    amrex::MultiFab* ux,
    amrex::MultiFab* lx,
    amrex::MultiFab* uy,
    amrex::MultiFab* ly,
    amrex::MultiFab* uz,
    amrex::MultiFab* lz,
    amrex::Geometry geom,
    const BcArray& bcArr,
    const SolverConfig& config
)
    : exec_(std::move(exec)), nexec_(nexec), onDevice_(exec_->get_master().get() != exec_.get()),
      n_(static_cast<gko::size_type>(alpha->boxArray().numPts()))
{
    if (onDevice_)
    {
        // The device residual kernel reads the caller's coefficients directly, so in-place
        // updates ARE seen -- this path derives its diagonal per apply, unlike FaceCoeffOp.
        alpha_ = alpha;
        ux_ = ux;
        lx_ = lx;
        uy_ = uy;
        ly_ = ly;
        uz_ = uz;
        lz_ = lz;
        bcData_ = config.bcData;
    }
    else
    {
        // Host residual loops can't read device memory: the coefficients are staged to pinned
        // ONCE here, so on this path an in-place caller update is NOT observed.
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
        if (config.bcData != nullptr)
        {
            ownedBcData_ = pinnedCopy(*config.bcData);
            bcData_ = ownedBcData_.get();
        }
    }
    geom_ = geom;
    bcArr_ = bcArr;
    hasPhysBc_ = std::any_of(bcArr.begin(), bcArr.end(), [](int b) { return b != 0; });
    maxIter_ = config.maxIter;
    rtol_ = config.rtol;
    atol_ = config.atol;
    // The stationary loop runs its own stopping test; KrylovSolver::build is not involved.
    norm_ = parseNorm(config.norm);
    projectNull_ = config.projectNullspace;
    // Non-owning handles onto the caller's fields, grouped (core/fieldLevel.hpp).
    GmgHierarchy h = buildGmgHierarchy(
        exec_,
        n_,
        CellFieldLevel {nonOwning(*alpha)},
        FaceFieldLevel {{nonOwning(*ux), nonOwning(*uy), nonOwning(*uz)}},
        FaceFieldLevel {{nonOwning(*lx), nonOwning(*ly), nonOwning(*lz)}},
        MeshLevel {alpha->boxArray(), alpha->DistributionMap(), geom},
        bcArr,
        config.precondCycles,
        config.gmg
    );
    gmgOwner_ = h.op;
    gmgMf_ = h.mf;
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
}

void GmgStationarySolver::fillGmgGhosts(amrex::MultiFab& mf) const
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
                fillDomainBcGhostsInhomDevice(nexec_, mf, *bcData_, geom_.Domain(), bcArr_, dx);
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

void GmgStationarySolver::subtractMeanMf(amrex::MultiFab& mf) const
{
    const double mean = mf.sum(0) / static_cast<double>(n_);
    mf.plus(-mean, 0, 1);
}

SolveResult GmgStationarySolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    // Warm start: x0 = incoming sol (do NOT zero — persistent-solver contract).
    amrex::MultiFab::Copy(*xWork_, sol, 0, 0, 1, 0);

    // Host residual loops can't read the device rhs: staged to pinned once per solve, since it
    // is constant across the cycle loop. The device path reads rhs directly.
    const amrex::MultiFab* rhsUse = &rhs;
    if (!onDevice_)
    {
        amrex::MultiFab::Copy(*rhsPinned_, rhs, 0, 0, 1, 0);
        amrex::Gpu::streamSynchronize();
        rhsUse = rhsPinned_.get();
    }

    // Stopping test in either norm: ||r|| <= max(rtol*||b||, atol), both measured in the same
    // norm (norm="linf" is MLMG's ||.||_inf, so a solve can be held to MLMG's own criterion).
    const bool useInf = (norm_ == NormKind::linf);
    const double bNorm = useInf ? rhs.norminf(0, 1, amrex::IntVect(0)) : rhs.norm2(0);
    const double stopTol = std::max(rtol_ * bNorm, atol_);
    const double rhsMean = projectNull_ ? rhs.sum(0) / static_cast<double>(n_) : 0.0;
    if (projectNull_)
    {
        subtractMeanMf(*xWork_);
    }

    std::vector<double> history;
    // One fused kernel forms the FP64 residual r = rhs - A x - rhsMean, casts it into the L0
    // rhs and reduces ||r|| in double; the nullspace shift folds into the same kernel.
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

    SolveResult out = makeSolveResult(static_cast<std::int64_t>(cycles), rnorm, converged, history);

    // Convergence diagnostic, reported rather than printed: a stationary V-cycle contracts by
    // a roughly constant factor per cycle, so `contraction` says whether the cycle is working
    // even on a run that converged. Threshold: report/blockamr-precision-measurements.md
    constexpr double slowRho = 0.464; // 10^(-1/3), i.e. one decade per 3 cycles
    const double rho = out.contraction.value_or(0.0);
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
    out.diagnostic = diagnostic;
    return out;
}

} // namespace

namespace
{

// Ginkgo Krylov solve of the matrix-free FaceCoeffOp (every solver except the native
// stationary loop above). All it adds over KrylovSolver is the inhomogeneous-BC refresh:
// FaceCoeffOp's c0 = L(0) is recomputed every solve so an in-place bc_data update takes effect.
class FaceCoeffKrylovSolver : public KrylovSolver
{
public:

    FaceCoeffKrylovSolver(
        std::shared_ptr<const gko::Executor> exec,
        const NeoN::Executor& nexec,
        amrex::MultiFab* alpha,
        amrex::MultiFab* ux,
        amrex::MultiFab* lx,
        amrex::MultiFab* uy,
        amrex::MultiFab* ly,
        amrex::MultiFab* uz,
        amrex::MultiFab* lz,
        amrex::Geometry geom,
        const BcArray& bcArr,
        const SolverConfig& config
    );

    SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) override;

private:

    const FaceCoeffOp* bcOffsetOp_ = nullptr;
};

FaceCoeffKrylovSolver::FaceCoeffKrylovSolver(
    std::shared_ptr<const gko::Executor> exec,
    const NeoN::Executor& nexec,
    amrex::MultiFab* alpha,
    amrex::MultiFab* ux,
    amrex::MultiFab* lx,
    amrex::MultiFab* uy,
    amrex::MultiFab* ly,
    amrex::MultiFab* uz,
    amrex::MultiFab* lz,
    amrex::Geometry geom,
    const BcArray& bcArr,
    const SolverConfig& config
)
    : KrylovSolver(
        exec, static_cast<gko::size_type>(alpha->boxArray().numPts()), localCount(*alpha)
    )
{
    // Non-owning handles: Python owns these fields and must outlive this solver.
    const MeshLevel mesh {alpha->boxArray(), alpha->DistributionMap(), geom};
    const CellFieldLevel alphaLevel {nonOwning(*alpha)};
    const FaceFieldLevel upper {{nonOwning(*ux), nonOwning(*uy), nonOwning(*uz)}};
    const FaceFieldLevel lower {{nonOwning(*lx), nonOwning(*ly), nonOwning(*lz)}};
    auto op = gko::share(
        FaceCoeffOp::create(exec_, nexec, mesh, n_, alphaLevel, upper, lower, bcArr, config.bcData)
    );
    if (config.bcData != nullptr)
    {
        // The typed hook solve() calls to refresh c0, plus the vector to hold
        // it. op_ (set by build() below) keeps the operator alive.
        bcOffsetOp_ = op.get();
        bcOffset_ = Dense::create(exec_, gko::dim<2> {nLocal_, 1});
    }

    // solver="ir": gko::solver::Ir<double> over the FaceCoeffOp above, with the generated GMG
    // V-cycle as its inner solver. Like solver="gmg" it implies the hierarchy and ignores
    // `precond`, but the loop runs through Ginkgo's Dense pack/unpack and does NOT fuse.
    if (config.solverKind == SolverKind::ir)
    {
        GmgHierarchy inner = buildGmgHierarchy(
            exec_, n_, alphaLevel, upper, lower, mesh, bcArr, config.precondCycles, config.gmg
        );
        build(
            op,
            config.solver,
            config.maxIter,
            config.rtol,
            config.atol,
            config.projectNullspace,
            std::move(inner.op),
            config.norm
        );
        return;
    }

    // The whole precond= fork -- gmg, gmg_kokkos, mlmg and their refusals -- lives in
    // makeFaceCoeffPrecond (precond.cpp), which la::Matrix also builds through.
    FaceCoeffPrecond built =
        makeFaceCoeffPrecond(exec_, n_, alphaLevel, upper, lower, mesh, bcArr, config);
    std::shared_ptr<const gko::LinOp> pc = std::move(built.op);
    // Set only by precond="gmg_kokkos"; solver="mpir" needs it and says so.
    std::shared_ptr<blockamr::KokkosGmgApply> vcycle = std::move(built.kokkosVcycle);
    // Mixed-precision refinement, expressed through the "ir" path: the OUTER loop stays
    // Ginkgo's Ir over the fp64 operator, so the residual, the stopping test and the answer
    // are the fp64 solver's; only the inner solver changes, to a preconditioned Cg<float>.
    std::string krylov = config.solver;
    if (config.solverKind == SolverKind::mpir)
    {
        if (!vcycle)
        {
            throw std::runtime_error(
                "FaceCoeffSolver: solver='mpir' needs precond='gmg_kokkos' (it is the only "
                "preconditioner with an fp32 apply)"
            );
        }
        auto op32 = gko::share(
            FaceCoeffOp32::create(exec_, nexec, mesh, n_, alphaLevel, upper, lower, bcArr)
        );
        auto pc32 = gko::share(GmgKokkosPrecond32::create(exec_, n_, vcycle));
        // l2 rather than the caller's norm: this is an INNER tolerance, not a
        // convergence claim, and ResidualNormInf is an fp64 criterion.
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> innerCriteria {
            gko::stop::Iteration::build()
                .with_max_iters(static_cast<gko::size_type>(config.mpInnerMaxIter))
                .on(exec_),
            gko::stop::ResidualNorm<float>::build()
                .with_baseline(gko::stop::mode::rhs_norm)
                .with_reduction_factor(static_cast<float>(config.mpInnerRtol))
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
    build(
        op,
        krylov,
        config.maxIter,
        config.rtol,
        config.atol,
        config.projectNullspace,
        std::move(pc),
        config.norm
    );
}

SolveResult FaceCoeffKrylovSolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    if (bcOffsetOp_ != nullptr)
    {
        // c0 = L(0), refreshed every solve: the BC data is REFERENCED, not copied, on the
        // device path. The COEFFICIENTS are not -- the diagonal is computed at construction, so
        // a coefficient change needs a new solver. x_ is the zero source (solve overwrites it).
        x_->fill(0.0);
        bcOffsetOp_->applyBcOffset(x_.get(), bcOffset_.get());
    }
    return KrylovSolver::solve(rhs, sol);
}

} // namespace

namespace
{

// Runs every configuration check once, before EITHER concrete solver allocates anything, then
// builds whichever ISolver the configuration calls for.
std::unique_ptr<ISolver> makeFaceCoeffSolver(
    std::shared_ptr<const gko::Executor> exec,
    const NeoN::Executor& nexec,
    amrex::MultiFab* alpha,
    amrex::MultiFab* ux,
    amrex::MultiFab* lx,
    amrex::MultiFab* uy,
    amrex::MultiFab* ly,
    amrex::MultiFab* uz,
    amrex::MultiFab* lz,
    amrex::Geometry geom,
    const SolverConfig& config
)
{
    // REFUSED rather than ignored: the shipped GmgPrecondT stores one type per level, so
    // accepting this would report a narrowed-coefficient timing for a hierarchy that is not.
    if (!config.gmg.coeffPrecision.empty() && config.precondKind != PrecondKind::gmg_kokkos)
    {
        throw std::runtime_error(
            "FaceCoeffSolver: gmg_coeff_precision needs precond='gmg_kokkos' (the shipped GMG "
            "hierarchy stores its coefficients in the same type as its fields)"
        );
    }

    // The V-cycle is SPD only with equal pre/post sweep counts; asymmetric counts break CG's
    // assumption, so warn but ALLOW (still usable as a stationary/flexible-CG smoother). The
    // native stationary solver is not CG, so asymmetric sweeps there are legitimate.
    if (config.precondKind == PrecondKind::gmg && config.solverKind == SolverKind::cg
        && config.gmg.preSweeps != config.gmg.postSweeps)
    {
        std::cerr << "FaceCoeffSolver: warning — gmg_pre_sweeps (" << config.gmg.preSweeps
                  << ") != gmg_post_sweeps (" << config.gmg.postSweeps
                  << ") makes the V-cycle non-symmetric; CG may stall or diverge. "
                     "Use equal counts for a CG-safe preconditioner.\n";
    }
    const BcArray bcArr = parseBc(config.bc, geom, "FaceCoeffSolver");
    if (config.bcData != nullptr)
    {
        checkBcData(*config.bcData, *alpha, bcArr, "FaceCoeffSolver");
    }

    forbidPrecondMlmg(config, config.solverKind == SolverKind::gmg, "solver='gmg'");
    forbidPrecondMlmg(config, config.solverKind == SolverKind::ir, "solver='ir'");
    forbidPrecondMlmg(config, config.precondKind == PrecondKind::gmg, "precond='gmg'");
    forbidPrecondMlmg(
        config, config.precondKind == PrecondKind::gmg_kokkos, "precond='gmg_kokkos'"
    );

    if (config.solverKind == SolverKind::gmg)
    {
        return std::make_unique<GmgStationarySolver>(
            exec, nexec, alpha, ux, lx, uy, ly, uz, lz, geom, bcArr, config
        );
    }
    return std::make_unique<FaceCoeffKrylovSolver>(
        exec, nexec, alpha, ux, lx, uy, ly, uz, lz, geom, bcArr, config
    );
}

} // namespace

FaceCoeffSolver::FaceCoeffSolver(
    const NeoN::Executor& executor,
    amrex::Geometry geom,
    amrex::MultiFab* alpha,
    amrex::MultiFab* ux,
    amrex::MultiFab* lx,
    amrex::MultiFab* uy,
    amrex::MultiFab* ly,
    amrex::MultiFab* uz,
    amrex::MultiFab* lz,
    const SolverConfig& config
)
    : impl_(makeFaceCoeffSolver(
        makeExecutor(executor), executor, alpha, ux, lx, uy, ly, uz, lz, geom, config
    ))
{}

SolveResult FaceCoeffSolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    return impl_->solve(rhs, sol);
}

namespace
{

// FaceCoeffCsrSolver has no matrix-free GMG hierarchy and no fp32 inner solve, so these 16
// knobs do nothing. REFUSED rather than ignored: accepting a knob that does nothing would
// report a configuration that was never applied.
void validateForCsr(const SolverConfig& config)
{
    static const GmgConfig kDefaultGmg {};
    std::vector<std::string> offending;
    if (config.gmg.preSweeps != kDefaultGmg.preSweeps) offending.push_back("gmg_pre_sweeps");
    if (config.gmg.postSweeps != kDefaultGmg.postSweeps) offending.push_back("gmg_post_sweeps");
    if (config.gmg.coarsestSweeps != kDefaultGmg.coarsestSweeps)
        offending.push_back("gmg_coarsest_sweeps");
    if (config.gmg.maxLevels != kDefaultGmg.maxLevels) offending.push_back("gmg_max_levels");
    if (config.gmg.minBottom != kDefaultGmg.minBottom) offending.push_back("gmg_min_bottom");
    if (config.gmg.smoother != kDefaultGmg.smoother) offending.push_back("gmg_smoother");
    if (config.gmg.precision != kDefaultGmg.precision) offending.push_back("gmg_precision");
    if (config.gmg.coeffPrecision != kDefaultGmg.coeffPrecision)
        offending.push_back("gmg_coeff_precision");
    if (config.gmg.omega != kDefaultGmg.omega) offending.push_back("gmg_omega");
    if (config.gmg.aggLevel0Size != kDefaultGmg.aggLevel0Size)
        offending.push_back("gmg_agg_l0_size");
    if (config.gmg.symmetric != kDefaultGmg.symmetric) offending.push_back("symmetric");
    if (config.gmg.bottomSolver != kDefaultGmg.bottomSolver)
        offending.push_back("gmg_bottom_solver");
    if (config.gmg.bottomMaxIter != kDefaultGmg.bottomMaxIter)
        offending.push_back("gmg_bottom_max_iter");
    if (config.gmg.bottomRtol != kDefaultGmg.bottomRtol) offending.push_back("gmg_bottom_rtol");
    static constexpr double kDefaultMpInnerRtol = 1e-2;
    static constexpr int kDefaultMpInnerMaxIter = 20;
    if (config.mpInnerRtol != kDefaultMpInnerRtol) offending.push_back("mp_inner_rtol");
    if (config.mpInnerMaxIter != kDefaultMpInnerMaxIter) offending.push_back("mp_inner_max_iter");
    if (offending.empty())
    {
        return;
    }
    std::string joined;
    for (std::size_t i = 0; i < offending.size(); ++i)
    {
        if (i) joined += ", ";
        joined += offending[i];
    }
    throw std::runtime_error(
        "FaceCoeffCsrSolver: " + joined
        + " only apply to the matrix-free GMG hierarchy (precond='gmg'/'gmg_kokkos' or "
          "solver='gmg'/'ir'/'mpir') and are not accepted by the assembled-CSR solver — use "
          "FaceCoeffSolver, or omit them to keep the default."
    );
}

} // namespace

FaceCoeffCsrSolver::FaceCoeffCsrSolver(
    const NeoN::Executor& executor,
    amrex::Geometry geom,
    amrex::MultiFab* alpha,
    amrex::MultiFab* ux,
    amrex::MultiFab* lx,
    amrex::MultiFab* uy,
    amrex::MultiFab* ly,
    amrex::MultiFab* uz,
    amrex::MultiFab* lz,
    const SolverConfig& config
)
    : KrylovSolver(
        makeExecutor(executor),
        static_cast<gko::size_type>(alpha->boxArray().numPts()),
        localCount(*alpha)
    )
{
    validateForCsr(config);
    // The assembly is single-box only (csr.cpp), which on >1 rank would mean one
    // rank holding every row while the others hold none.
    requireSingleRank("FaceCoeffCsrSolver");
    // Periodic sides keep their wraparound column; homogeneous dirichlet/neumann sides are
    // folded onto the diagonal by assembleFaceCoeffCsr, the assembled twin of FaceCoeffOp's
    // ghost reflection. parseBc rejects a bc that disagrees with the geometry's periodicity.
    const BcArray bcArr = parseBc(config.bc, geom, "FaceCoeffCsrSolver");
    // bc_data stays REFUSED: inhomogeneous BCs are an affine term L(x) = A x + c0, i.e. an rhs
    // fold rather than a matrix one, and this path has no c0. A silently-dropped datum would
    // look like a wrong answer instead of a missing feature.
    if (config.bcData != nullptr)
    {
        throw std::runtime_error(
            "FaceCoeffCsrSolver: bc_data is not supported — the assembled matrix folds "
            "homogeneous dirichlet/neumann and periodic boundaries only; inhomogeneous BC "
            "data needs FaceCoeffSolver"
        );
    }
    // CSR's own, narrower combination legality (the spelling was validated at the boundary):
    // with no matrix-free operator, 'gmg' and 'gmg_kokkos' are REFUSED.
    if (config.precondKind == PrecondKind::gmg)
    {
        throw std::runtime_error(
            "FaceCoeffCsrSolver: precond='gmg' is matrix-free only — use FaceCoeffSolver"
        );
    }
    if (config.precondKind != PrecondKind::none && config.precondKind != PrecondKind::mlmg)
    {
        throw std::runtime_error(
            "FaceCoeffCsrSolver: unknown precond '" + config.precond
            + "' (expected 'none' or 'mlmg')"
        );
    }
    if (config.precondKind == PrecondKind::mlmg && config.precondMlmg == nullptr)
    {
        throw std::runtime_error("FaceCoeffCsrSolver: precond='mlmg' requires precond_mlmg");
    }
    // Non-owning handles onto the caller's fields, grouped (core/fieldLevel.hpp).
    auto op = assembleFaceCoeffCsr(
        exec_,
        MeshLevel {alpha->boxArray(), alpha->DistributionMap(), geom},
        CellFieldLevel {nonOwning(*alpha)},
        FaceFieldLevel {{nonOwning(*ux), nonOwning(*uy), nonOwning(*uz)}},
        FaceFieldLevel {{nonOwning(*lx), nonOwning(*ly), nonOwning(*lz)}},
        bcArr
    );
    // Null when there is no precond_mlmg to wrap (precond.cpp).
    std::shared_ptr<const gko::LinOp> pc = makeMlmgPrecond(exec_, n_, *alpha, config);
    build(
        op,
        config.solver,
        config.maxIter,
        config.rtol,
        config.atol,
        config.projectNullspace,
        std::move(pc),
        config.norm
    );
}

} // namespace blockamr::la
