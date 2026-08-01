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
#include "NeoN/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/mlmgOps.hpp"

namespace blockamr::la
{

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

namespace
{

// The device path syncs ONCE, after both gathers.
void packSystem(
    bool onDevice, const amrex::MultiFab& rhs, const amrex::MultiFab& sol, double* b, double* x
)
{
    prof::Timer t("solve.pack");
    if (!onDevice)
    {
        gather(rhs, b, 1.0);
        gather(sol, x, 1.0);
        return;
    }
    gather_device(rhs, b, 1.0);
    gather_device(sol, x, 1.0);
    amrex::Gpu::streamSynchronize();
}

// The device path drains Ginkgo's queue before AMReX reads the buffer.
void unpackSolution(bool onDevice, const gko::Executor& exec, const double* x, amrex::MultiFab& sol)
{
    prof::Timer t("solve.unpack");
    if (!onDevice)
    {
        scatter(x, sol);
        return;
    }
    exec.synchronize();
    scatter_device(x, sol);
    amrex::Gpu::streamSynchronize();
}

// ||b - A x|| in the norm the solve stopped on, so res_norm is comparable with the rtol that
// produced it. Takes the GLOBAL views, so the norm reduces across ranks as the criterion did.
double finalResidualNorm(
    const gko::LinOp& op,
    const std::shared_ptr<gko::LinOp>& b,
    const std::shared_ptr<gko::LinOp>& x,
    NormKind norm,
    const std::shared_ptr<const gko::Executor>& exec
)
{
    prof::Timer t("solve.report");
    auto res = b->clone();
    auto one = gko::initialize<Dense>({1.0}, exec);
    auto negOne = gko::initialize<Dense>({-1.0}, exec);
    op.apply(negOne, x, one, res);
    return (norm == NormKind::linf) ? normInf(res.get()) : globalNorm2(res.get());
}

} // namespace

SolveResult KrylovSolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    resLogger_->clear(); // per-call history
    packSystem(onDevice_, rhs, sol, b_->get_values(), x_->get_values());

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

    unpackSolution(onDevice_, *exec_, x_->get_const_values(), sol);

    const double resNorm = finalResidualNorm(*op_, bGlobal_, xGlobal_, norm_, exec_);
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
    const SolverConfig& config,
    std::shared_ptr<const gko::LinOp> precond
)
{
    norm_ = parseNorm(config.norm);
    op_ = std::move(op);
    // rhs_norm: the rtol is RELATIVE here, recomputed per solve from the incoming right-hand
    // side, which is what lets one generate() serve many of them.
    const StopSpec stop {
        config.maxIter, gko::stop::mode::rhs_norm, config.rtol, config.atol, config.norm
    };
    solver_ = buildKrylov(solver, exec_, op_, stop, std::move(precond));
    logger_ = gko::share(gko::log::Convergence<double>::create());
    solver_->add_logger(logger_);
    resLogger_ = std::make_shared<ResidualHistoryLogger>();
    solver_->add_logger(resLogger_);
    projectNullspace_ = config.projectNullspace;
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

// Everything the two concrete solvers below are built from is la::FaceCoeffLevel
// (linearAlgebra/precond.hpp), which the hierarchy builders take too.

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
// linearAlgebra/precond.cpp so la::makeHierarchy builds the same one (precond.hpp).

namespace
{

// Native stationary geometric-multigrid solver (solver="gmg"): x <- x + V(b - A x) run to
// tolerance, the whole loop on AMReX fabs with no Ginkgo Krylov object or Dense vector --
// which is why it does not derive from KrylovSolver.
class GmgStationarySolver : public ISolver
{
public:

    GmgStationarySolver(
        std::shared_ptr<const gko::Executor> exec,
        const NeoN::Executor& nexec,
        const FaceCoeffLevel& level,
        const BcArray& bcArr,
        const SolverConfig& config
    );

    SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) override;

private:

    // What one run of the stationary loop ended on.
    struct CycleOutcome
    {
        int cycles;
        double rnorm;
        bool converged;
    };

    // Point the coefficient members at the caller's fields, or at pinned copies of them.
    void stageCoefficients(const FaceCoeffLevel& level, const amrex::MultiFab* bcData);

    // Allocate xWork_ (and, on the host path, rhsPinned_).
    void allocateWorkFabs(const MeshLevel& mesh);

    // rhs as the residual loop may read it: the caller's fab, or a pinned stage of it.
    const amrex::MultiFab& stageRhs(const amrex::MultiFab& rhs) const;

    // x <- x + V(rhs - A x - rhsMean) until ||r|| <= stopTol or maxIter_ cycles, appending one
    // residual norm per evaluation to `history` (the first being the initial residual).
    CycleOutcome runCycles(
        const amrex::MultiFab& rhsUse, double rhsMean, double stopTol, std::vector<double>& history
    ) const;

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

void GmgStationarySolver::stageCoefficients(
    const FaceCoeffLevel& level, const amrex::MultiFab* bcData
)
{
    if (onDevice_)
    {
        // The device residual kernel reads the caller's coefficients directly, so in-place
        // updates ARE seen -- this path derives its diagonal per apply, unlike FaceCoeffOp.
        alpha_ = &*level.alpha;
        ux_ = &level.upper[0];
        lx_ = &level.lower[0];
        uy_ = &level.upper[1];
        ly_ = &level.lower[1];
        uz_ = &level.upper[2];
        lz_ = &level.lower[2];
        bcData_ = bcData;
        return;
    }

    // Host residual loops can't read device memory: the coefficients are staged to pinned
    // ONCE here, so on this path an in-place caller update is NOT observed.
    ownedCoeff_ = {
        pinnedCopy(*level.alpha),
        pinnedCopy(level.upper[0]),
        pinnedCopy(level.lower[0]),
        pinnedCopy(level.upper[1]),
        pinnedCopy(level.lower[1]),
        pinnedCopy(level.upper[2]),
        pinnedCopy(level.lower[2])
    };
    alpha_ = ownedCoeff_[0].get();
    ux_ = ownedCoeff_[1].get();
    lx_ = ownedCoeff_[2].get();
    uy_ = ownedCoeff_[3].get();
    ly_ = ownedCoeff_[4].get();
    uz_ = ownedCoeff_[5].get();
    lz_ = ownedCoeff_[6].get();
    if (bcData != nullptr)
    {
        ownedBcData_ = pinnedCopy(*bcData);
        bcData_ = ownedBcData_.get();
    }
}

void GmgStationarySolver::allocateWorkFabs(const MeshLevel& mesh)
{
    if (onDevice_)
    {
        xWork_ = std::make_shared<amrex::MultiFab>(mesh.ba, mesh.dm, 1, 1);
        return;
    }
    xWork_ = std::make_shared<amrex::MultiFab>(
        mesh.ba, mesh.dm, 1, 1, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
    );
    rhsPinned_ = std::make_shared<amrex::MultiFab>(
        mesh.ba, mesh.dm, 1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
    );
}

GmgStationarySolver::GmgStationarySolver(
    std::shared_ptr<const gko::Executor> exec,
    const NeoN::Executor& nexec,
    const FaceCoeffLevel& level,
    const BcArray& bcArr,
    const SolverConfig& config
)
    : exec_(std::move(exec)), nexec_(nexec), onDevice_(exec_->get_master().get() != exec_.get()),
      n_(static_cast<gko::size_type>(level.mesh.ba.numPts())), geom_(level.mesh.geom),
      bcArr_(bcArr),
      hasPhysBc_(std::any_of(bcArr.begin(), bcArr.end(), [](int b) { return b != 0; })),
      maxIter_(config.maxIter), rtol_(config.rtol), atol_(config.atol),
      projectNull_(config.projectNullspace),
      // The stationary loop runs its own stopping test; KrylovSolver::build is not involved.
      norm_(parseNorm(config.norm))
{
    stageCoefficients(level, config.bcData);
    GmgHierarchy h = buildGmgHierarchy(exec_, n_, level, bcArr, config);
    gmgOwner_ = h.op;
    gmgMf_ = h.mf;
    allocateWorkFabs(level.mesh);
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

const amrex::MultiFab& GmgStationarySolver::stageRhs(const amrex::MultiFab& rhs) const
{
    // Host residual loops can't read the device rhs: staged to pinned once per solve, since it
    // is constant across the cycle loop. The device path reads rhs directly.
    if (onDevice_)
    {
        return rhs;
    }
    amrex::MultiFab::Copy(*rhsPinned_, rhs, 0, 0, 1, 0);
    amrex::Gpu::streamSynchronize();
    return *rhsPinned_;
}

GmgStationarySolver::CycleOutcome GmgStationarySolver::runCycles(
    const amrex::MultiFab& rhsUse, double rhsMean, double stopTol, std::vector<double>& history
) const
{
    const bool useInf = (norm_ == NormKind::linf);
    // One fused kernel forms the FP64 residual r = rhs - A x - rhsMean, casts it into the L0
    // rhs and reduces ||r|| in double; the nullspace shift folds into the same kernel.
    auto computeResid = [&]() -> double
    {
        prof::Timer t("gmg.solve.resid");
        fillGmgGhosts(*xWork_);
        const ResidNorms nr = gmgMf_->residScatterNorm(GmgResidualInput {
            .sol = xWork_.get(),
            .rhs = &rhsUse,
            .alpha = alpha_,
            .ux = ux_,
            .lx = lx_,
            .uy = uy_,
            .ly = ly_,
            .uz = uz_,
            .lz = lz_,
            .shift = rhsMean
        });
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
    return {cycles, rnorm, converged};
}

// Reported rather than printed: a stationary V-cycle contracts by a roughly constant factor per
// cycle, so `rho` says whether the cycle is working even on a run that converged.
// Threshold: report/blockamr-precision-measurements.md
const char* contractionDiagnostic(int cycles, double rho)
{
    constexpr double slowRho = 0.464; // 10^(-1/3), i.e. one decade per 3 cycles
    if (cycles > 0 && rho >= 1.0)
    {
        return "V-cycle is not contracting (residual grew or stalled). Check the "
               "bottom solve (gmg_bottom_solver), cell aspect ratio and coefficient "
               "contrast.";
    }
    if (cycles > 1 && rho > slowRho)
    {
        return "V-cycle is contracting slowly (worse than one decade per 3 cycles). "
               "The usual cause is a bottom grid too large for gmg_coarsest_sweeps -- "
               "try gmg_bottom_solver='cg' (or 'bicgstab' when symmetric=False).";
    }
    return "";
}

SolveResult GmgStationarySolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    // Warm start: x0 = incoming sol (do NOT zero — persistent-solver contract).
    amrex::MultiFab::Copy(*xWork_, sol, 0, 0, 1, 0);
    const amrex::MultiFab& rhsUse = stageRhs(rhs);

    // Stopping test in either norm: ||r|| <= max(rtol*||b||, atol), both measured in the same
    // norm (norm="linf" is MLMG's ||.||_inf, so a solve can be held to MLMG's own criterion).
    const double bNorm =
        (norm_ == NormKind::linf) ? rhs.norminf(0, 1, amrex::IntVect(0)) : rhs.norm2(0);
    const double stopTol = std::max(rtol_ * bNorm, atol_);
    const double rhsMean = projectNull_ ? rhs.sum(0) / static_cast<double>(n_) : 0.0;
    if (projectNull_)
    {
        subtractMeanMf(*xWork_);
    }

    std::vector<double> history;
    const CycleOutcome outcome = runCycles(rhsUse, rhsMean, stopTol, history);

    amrex::MultiFab::Copy(sol, *xWork_, 0, 0, 1, 0);

    SolveResult out = makeSolveResult(
        static_cast<std::int64_t>(outcome.cycles), outcome.rnorm, outcome.converged, history
    );
    out.diagnostic = contractionDiagnostic(outcome.cycles, out.contraction.value_or(0.0));
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
        const FaceCoeffLevel& level,
        const BcArray& bcArr,
        const SolverConfig& config
    );

    SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) override;

private:

    // Mixed-precision refinement (solver="mpir"), expressed as the inner solver of the "ir"
    // path: the OUTER loop stays Ginkgo's Ir over the fp64 operator, so the residual, the
    // stopping test and the answer are the fp64 solver's; only this inner solver is fp32.
    // `vcycle` is precond="gmg_kokkos"'s own V-cycle; throws when it is null.
    std::shared_ptr<const gko::LinOp> makeMixedPrecisionInner(
        const NeoN::Executor& nexec,
        const FaceCoeffLevel& level,
        const BcArray& bcArr,
        const SolverConfig& config,
        const std::shared_ptr<blockamr::KokkosGmgApply>& vcycle
    );

    // The GMG V-cycle this level implies, as solver="ir" wants it: an inner solver.
    std::shared_ptr<const gko::LinOp>
    makeInnerGmgOp(const FaceCoeffLevel& level, const BcArray& bcArr, const SolverConfig& config);

    const FaceCoeffOp* bcOffsetOp_ = nullptr;
};

std::shared_ptr<const gko::LinOp> FaceCoeffKrylovSolver::makeInnerGmgOp(
    const FaceCoeffLevel& level, const BcArray& bcArr, const SolverConfig& config
)
{
    GmgHierarchy inner = buildGmgHierarchy(exec_, n_, level, bcArr, config);
    return std::move(inner.op);
}

std::shared_ptr<const gko::LinOp> FaceCoeffKrylovSolver::makeMixedPrecisionInner(
    const NeoN::Executor& nexec,
    const FaceCoeffLevel& level,
    const BcArray& bcArr,
    const SolverConfig& config,
    const std::shared_ptr<blockamr::KokkosGmgApply>& vcycle
)
{
    if (!vcycle)
    {
        throw std::runtime_error(
            "FaceCoeffSolver: solver='mpir' needs precond='gmg_kokkos' (it is the only "
            "preconditioner with an fp32 apply)"
        );
    }
    auto op32 = gko::share(FaceCoeffOp32::create(exec_, nexec, level, DomainBc {bcArr}));
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
    return gko::share(MixedPrecisionSolve::create(exec_, n_, cg32));
}

FaceCoeffKrylovSolver::FaceCoeffKrylovSolver(
    std::shared_ptr<const gko::Executor> exec,
    const NeoN::Executor& nexec,
    const FaceCoeffLevel& level,
    const BcArray& bcArr,
    const SolverConfig& config
)
    : KrylovSolver(
        exec, static_cast<gko::size_type>(level.mesh.ba.numPts()), localCount(*level.alpha)
    )
{
    auto op = gko::share(FaceCoeffOp::create(exec_, nexec, level, DomainBc {bcArr, config.bcData}));
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
        build(op, config.solver, config, makeInnerGmgOp(level, bcArr, config));
        return;
    }

    // The whole precond= fork -- gmg, gmg_kokkos, mlmg and their refusals -- lives in
    // makeFaceCoeffPrecond (precond.cpp), which la::makeHierarchy also builds through.
    FaceCoeffPrecond built = makeFaceCoeffPrecond(exec_, n_, level, bcArr, config);
    std::shared_ptr<const gko::LinOp> pc = std::move(built.op);
    std::string krylov = config.solver;
    if (config.solverKind == SolverKind::mpir)
    {
        pc = makeMixedPrecisionInner(nexec, level, bcArr, config, built.kokkosVcycle);
        krylov = "ir";
    }
    build(op, krylov, config, std::move(pc));
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

// The refusals that depend on the configuration alone, run before either concrete solver
// allocates anything.
void checkSolverConfig(const SolverConfig& config)
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
}

// Runs every configuration check once, before EITHER concrete solver allocates anything, then
// builds whichever ISolver the configuration calls for.
std::unique_ptr<ISolver> makeFaceCoeffSolver(
    std::shared_ptr<const gko::Executor> exec,
    const NeoN::Executor& nexec,
    const FaceCoeffLevel& level,
    const SolverConfig& config
)
{
    checkSolverConfig(config);
    const BcArray bcArr = parseBc(config.bc, level.mesh.geom, "FaceCoeffSolver");
    if (config.bcData != nullptr)
    {
        checkBcData(*config.bcData, *level.alpha, bcArr, "FaceCoeffSolver");
    }

    forbidPrecondMlmg(config, config.solverKind == SolverKind::gmg, "solver='gmg'");
    forbidPrecondMlmg(config, config.solverKind == SolverKind::ir, "solver='ir'");
    forbidPrecondMlmg(config, config.precondKind == PrecondKind::gmg, "precond='gmg'");
    forbidPrecondMlmg(
        config, config.precondKind == PrecondKind::gmg_kokkos, "precond='gmg_kokkos'"
    );

    if (config.solverKind == SolverKind::gmg)
    {
        return std::make_unique<GmgStationarySolver>(exec, nexec, level, bcArr, config);
    }
    return std::make_unique<FaceCoeffKrylovSolver>(exec, nexec, level, bcArr, config);
}

} // namespace

FaceCoeffSolver::FaceCoeffSolver(
    const NeoN::Executor& executor, const FaceCoeffLevel& level, const SolverConfig& config
)
    // A pure forward: the seven loose field pointers -- and the ux/uy/uz order that pairs them
    // up -- are now named once, in the binding that owns them (bindings/linearAlgebra.cpp).
    : impl_(makeFaceCoeffSolver(makeExecutor(executor), executor, level, config))
{}

SolveResult FaceCoeffSolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    return impl_->solve(rhs, sol);
}

} // namespace blockamr::la
