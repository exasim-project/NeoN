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
#include "NeoN/blockAmr/core/profiling.hpp"
#include "NeoN/blockAmr/core/transfer.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/gmgPrecond.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/precond.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/logging.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/mixedPrecision.hpp"
#include "NeoN/blockAmr/operators/csr.hpp"
#include "NeoN/blockAmr/operators/faceCoeffOp.hpp"
#include "NeoN/blockAmr/operators/mlmgOps.hpp"

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

// Declared in distVec.hpp. The two bodies are spelled out rather than shared
// through a template helper because nvcc rejects ANY template signature that
// returns shared_ptr<LinOp> here (see the note in distVec.hpp); the
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

// The one shape shared by all four "gmg forbids precond_mlmg" checks
// (solver='gmg'/'ir', precond='gmg'/'gmg_kokkos'): a V-cycle already IS the
// preconditioner/solver, so combining it with an externally-built MLMG makes
// no sense. `active` is the caller's own combination test (solverKind==gmg,
// etc.); `what` names which one fired, matching the four historical messages
// this replaces exactly.
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

namespace
{

// The hierarchy the native-GMG paths share: the stationary solver
// (GmgStationarySolver, below) drives it directly; the Krylov paths use it
// as an IR inner solver (solver="ir") or a preconditioner (precond="gmg").
// One definition, three call sites -- T9's eventual FaceCoeffs<T> bundle
// folds into this signature, not before.
struct GmgHierarchy
{
    std::shared_ptr<const gko::LinOp> op;
    const GmgApplyMf* mf = nullptr; // only read by the stationary path
};

GmgHierarchy buildGmgHierarchy(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    const amrex::Geometry& geom,
    const BcArray& bcArr,
    int precondCycles,
    const GmgConfig& gmg
)
{
    // bf16 is named separately from an outright typo: it exists, but only for
    // precond='gmg_kokkos'. The shipped GmgPrecondT hierarchy is fp64/fp32, and
    // instantiating it for a storage-only type would mean porting its Chebyshev
    // smoother and lambda-max power iteration too.
    if (gmg.precision == "bf16")
    {
        throw std::runtime_error("FaceCoeffSolver: gmg_precision='bf16' needs precond='gmg_kokkos' "
                                 "(the shipped GMG hierarchy is fp64/fp32 only)");
    }
    if (gmg.precision != "fp64" && gmg.precision != "fp32")
    {
        throw std::runtime_error(
            "FaceCoeffSolver: unknown gmg_precision '" + gmg.precision
            + "' (expected 'fp64' or 'fp32')"
        );
    }
    auto makeGmg = [&](auto tag) -> GmgHierarchy
    {
        using T = decltype(tag);
        auto p = GmgPrecondT<T>::create(
            exec,
            alpha->boxArray(),
            alpha->DistributionMap(),
            geom,
            n,
            alpha,
            ux,
            lx,
            uy,
            ly,
            uz,
            lz,
            bcArr,
            precondCycles,
            gmg.preSweeps,
            gmg.postSweeps,
            gmg.coarsestSweeps,
            gmg.maxLevels,
            gmg.minBottom,
            gmg.smoother,
            gmg.omega,
            gmg.symmetric,
            gmg.bottomSolver,
            gmg.bottomMaxIter,
            gmg.bottomRtol
        );
        GmgHierarchy h;
        h.mf = p.get(); // GmgPrecondT<T>* -> const GmgApplyMf* (kept alive by h.op below)
        h.op = gko::share(std::move(p));
        return h;
    };
    return (gmg.precision == "fp32") ? makeGmg(float {}) : makeGmg(double {});
}

} // namespace

namespace
{

// Native stationary geometric-multigrid solver (solver="gmg"): x <- x +
// V(b - A x), run to tolerance (Richardson iteration, like MLMG) -- no
// Ginkgo Krylov object, the whole loop on AMReX fabs. Self-contained: unlike
// the Krylov path this never touches a Ginkgo Dense vector, so it does not
// derive from KrylovSolver -- that is the whole point of splitting it out
// (KrylovSolver's base ctor used to need a config-derived allocDense=false
// to skip work this class never needed in the first place).
class GmgStationarySolver : public ISolver
{
public:

    GmgStationarySolver(
        std::shared_ptr<const gko::Executor> exec,
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        amrex::Geometry geom,
        const BcArray& bcArr,
        const SolverConfig& config
    );

    SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) override;

private:

    // Fill xWork_'s ghost layer for the FP64 residual: periodic/internal via
    // FillBoundary, then domain BCs via ghost reflection -- the same fill
    // FaceCoeffOp does, so the residual uses the identical operator A.
    //
    // With bcData_ the reflection is the INHOMOGENEOUS one, which makes the outer
    // residual rhs - L(x) rather than rhs - A x. That is the whole of the
    // stationary path's inhomogeneous-BC support: the V-cycle then solves
    // A delta = rhs - L(x) with its own homogeneous fills, which is right because
    // a correction's boundary condition is homogeneous whatever the solution's
    // is, and the iteration converges to L(x) = rhs. No extra apply, no rhs fold
    // — the Krylov path needs both only because Ginkgo requires a linear operator.
    void fillGmgGhosts(amrex::MultiFab& mf) const;

    // mf -= mean(mf) over the valid region (constant-nullspace projection for
    // singular systems; uniform cells so the volume mean is the arithmetic mean).
    void subtractMeanMf(amrex::MultiFab& mf) const;

    std::shared_ptr<const gko::Executor> exec_;
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
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    amrex::Geometry geom,
    const BcArray& bcArr,
    const SolverConfig& config
)
    : exec_(std::move(exec)), onDevice_(exec_->get_master().get() != exec_.get()),
      n_(static_cast<gko::size_type>(alpha->boxArray().numPts()))
{
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
        bcData_ = config.bcData;
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
    // The stationary loop runs its own stopping test, so KrylovSolver::build --
    // where the Krylov path records the norm -- is never involved here.
    norm_ = parseNorm(config.norm);
    projectNull_ = config.projectNullspace;
    GmgHierarchy h = buildGmgHierarchy(
        exec_, n_, alpha, ux, lx, uy, ly, uz, lz, geom, bcArr, config.precondCycles, config.gmg
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

void GmgStationarySolver::subtractMeanMf(amrex::MultiFab& mf) const
{
    const double mean = mf.sum(0) / static_cast<double>(n_);
    mf.plus(-mean, 0, 1);
}

SolveResult GmgStationarySolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
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
    // be held to exactly MLMG's criterion -- see stopNormInf.hpp).
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

    SolveResult out = makeSolveResult(static_cast<std::int64_t>(cycles), rnorm, converged, history);

    // Convergence diagnostic. A stationary V-cycle contracts the residual by a
    // roughly CONSTANT factor per cycle, so makeSolveResult's `contraction` (the
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

// Ginkgo Krylov solve of the matrix-free FaceCoeffOp (every solver="gmg"
// EXCEPT the native stationary loop above: "cg"/"bicgstab"/"gmres"/"gcr"/
// "fcg"/"ir"/"mpir", each optionally GMG-preconditioned). The only thing this
// adds over KrylovSolver is the inhomogeneous-BC refresh: FaceCoeffOp's c0 =
// L(0) has to be recomputed every solve so an in-place bc_data update takes
// effect, exactly as an in-place coefficient update does.
class FaceCoeffKrylovSolver : public KrylovSolver
{
public:

    FaceCoeffKrylovSolver(
        std::shared_ptr<const gko::Executor> exec,
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
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
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    amrex::Geometry geom,
    const BcArray& bcArr,
    const SolverConfig& config
)
    : KrylovSolver(
        exec, static_cast<gko::size_type>(alpha->boxArray().numPts()), localCount(*alpha)
    )
{
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
        config.bcData
    ));
    if (config.bcData != nullptr)
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
    if (config.solverKind == SolverKind::ir)
    {
        GmgHierarchy inner = buildGmgHierarchy(
            exec_, n_, alpha, ux, lx, uy, ly, uz, lz, geom, bcArr, config.precondCycles, config.gmg
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

    std::shared_ptr<const gko::LinOp> pc;
    // Set only by precond="gmg_kokkos"; solver="mpir" needs it and says so.
    std::shared_ptr<bench::KokkosGmgApply> vcycle;
    if (config.precondKind == PrecondKind::gmg)
    {
        pc = buildGmgHierarchy(
                 exec_,
                 n_,
                 alpha,
                 ux,
                 lx,
                 uy,
                 ly,
                 uz,
                 lz,
                 geom,
                 bcArr,
                 config.precondCycles,
                 config.gmg
        )
                 .op;
    }
    else if (config.precondKind == PrecondKind::gmg_kokkos)
    {
        // The same V-cycle as precond="gmg", under the optimised Kokkos launchers
        // (gmgKokkos/apply.hpp). A separate object rather than a mode of GmgPrecondT:
        // that one is the shipped baseline and stays untouched, so both can run in
        // one process and be compared directly.
        // Refused rather than ignored, for the same reason every other
        // capability gap on this path is: accepting a knob that does nothing
        // reports a Krylov bottom in the configuration and runs fixed sweeps.
        // The ported V-cycle lives behind the bench fence and has no Ginkgo, so
        // GmgBottomOp cannot reach it; closing this means porting the bottom
        // solve to that side, not relaxing the check.
        if (config.gmg.bottomSolver != "smoother")
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' has no Krylov bottom solve, so "
                "gmg_bottom_solver='"
                + config.gmg.bottomSolver
                + "' would silently run gmg_coarsest_sweeps sweeps. Use "
                  "precond='gmg' for a Krylov bottom."
            );
        }
        // The Kokkos V-cycle carries the same symmetry assumptions the shipped one
        // does (an over-relaxed red-black sweep, a self-adjoint cycle), and has no
        // path that would honour symmetric=False.
        if (!config.gmg.symmetric)
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' assumes a symmetric operator; "
                "symmetric=False needs precond='gmg'"
            );
        }
        if (config.gmg.smoother != "rbgs")
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' has only the red-black smoother, not '"
                + config.gmg.smoother + "'"
            );
        }
        bench::KokkosGmgOpts opts;
        opts.cycles = config.precondCycles;
        opts.preSweeps = config.gmg.preSweeps;
        opts.postSweeps = config.gmg.postSweeps;
        opts.coarsestSweeps = config.gmg.coarsestSweeps;
        opts.maxLevels = config.gmg.maxLevels;
        opts.minBottom = config.gmg.minBottom;
        opts.omega = config.gmg.omega;
        // Straight through, unvalidated here: makeKokkosGmgApply parses it and
        // throws on an unknown spelling, so a typo cannot quietly run fp64. This
        // is the only precond that has a bf16 hierarchy.
        opts.precision = config.gmg.precision;
        // Likewise unvalidated here beyond the guard above: makeKokkosGmgApply
        // rejects an unknown spelling and a coefficient type wider than the fields.
        opts.coeffPrecision = config.gmg.coeffPrecision;
        // The parsed spec straight through: the ported V-cycle carries the same
        // homogeneous Dirichlet/Neumann reflection as precond="gmg", built once per
        // level as a device plan rather than as a per-box AMReX launch.
        opts.bc = bcArr;
        opts.aggLevel0Size = config.gmg.aggLevel0Size;
        // Held in a local as well: solver="mpir" wraps the SAME hierarchy in an fp32
        // LinOp, and building it twice would double the setup and the device memory
        // for two views of one V-cycle.
        vcycle = std::shared_ptr<bench::KokkosGmgApply>(
            bench::makeKokkosGmgApply(geom, *alpha, *ux, *lx, *uy, *ly, *uz, *lz, opts)
        );
        pc = gko::share(GmgKokkosPrecond::create(exec_, n_, vcycle));
    }
    else
    {
        // config.precondKind is one of {none, mlmg, gmg, gmg_kokkos}
        // (parseSolverConfig already rejected anything else), and gmg/
        // gmg_kokkos are handled by the two branches above, so this is
        // precond="none"/"mlmg".
        // precond_mlmg alone implies "mlmg" (pre-existing behaviour).
        if (config.precondKind == PrecondKind::mlmg && config.precondMlmg == nullptr)
        {
            throw std::runtime_error("FaceCoeffSolver: precond='mlmg' requires precond_mlmg");
        }
        if (config.precondMlmg != nullptr)
        {
            pc = gko::share(MlmgPrecond::create(
                exec_,
                config.precondMlmg,
                alpha->boxArray(),
                alpha->DistributionMap(),
                n_,
                config.precondCycles
            ));
        }
    }
    // Mixed-precision iterative refinement. The OUTER loop is Ginkgo's Ir over the
    // fp64 operator -- it forms r = b - A x and runs the stopping test in fp64, so
    // the answer and the tolerance are the fp64 solver's -- and the inner correction
    // is a preconditioned Cg<float>. Expressed through the existing "ir" path
    // because Ir::with_generated_solver is exactly the hook needed: what changes is
    // only WHICH LinOp plays the inner solver.
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
        // c0 = L(0), refreshed every solve: the BC data is REFERENCED, not copied,
        // on the device path, so an in-place update has to take effect exactly as
        // an in-place coefficient update does. One extra operator apply per solve,
        // which is the whole price of inhomogeneous BCs on the Krylov path.
        //
        // x_ is the zero source rather than a dedicated vector: KrylovSolver::
        // solve overwrites it with the initial guess as its first act, so the fold
        // costs one n-vector (bcOffset_) instead of two.
        x_->fill(0.0);
        bcOffsetOp_->applyBcOffset(x_.get(), bcOffset_.get());
    }
    return KrylovSolver::solve(rhs, sol);
}

} // namespace

namespace
{

// Runs every check FaceCoeffSolver's constructor used to run unconditionally
// -- before the solver="gmg" vs. everything-else fork -- exactly once, then
// builds whichever concrete ISolver the configuration calls for. Order
// matches today's single constructor body exactly, since two of the checks
// (forbidPrecondMlmg's solver="ir"/precond="gmg"/"gmg_kokkos" cases) used to
// run BEFORE the matrix-free operator was built on the Krylov path -- moving
// them here keeps them running before EITHER concrete solver allocates
// anything.
std::unique_ptr<ISolver> makeFaceCoeffSolver(
    std::shared_ptr<const gko::Executor> exec,
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    amrex::Geometry geom,
    const SolverConfig& config
)
{
    // A separate coefficient precision exists in the Kokkos hierarchy alone. Named
    // rather than ignored: the shipped GmgPrecondT stores one type per level, so
    // accepting the option there would report a narrowed-coefficient timing for a
    // hierarchy that never narrowed anything.
    if (!config.gmg.coeffPrecision.empty() && config.precondKind != PrecondKind::gmg_kokkos)
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

    // "gmg forbids precond_mlmg", in all four modes that build a GMG hierarchy
    // (the native stationary loop, its Ginkgo-IR twin, and precond="gmg"/
    // "gmg_kokkos"): the V-cycle already IS the preconditioner/solver, so an
    // externally-built precond_mlmg would have nothing to do. Checked once,
    // here, for all four -- before either concrete solver allocates anything.
    forbidPrecondMlmg(config, config.solverKind == SolverKind::gmg, "solver='gmg'");
    forbidPrecondMlmg(config, config.solverKind == SolverKind::ir, "solver='ir'");
    forbidPrecondMlmg(config, config.precondKind == PrecondKind::gmg, "precond='gmg'");
    forbidPrecondMlmg(
        config, config.precondKind == PrecondKind::gmg_kokkos, "precond='gmg_kokkos'"
    );

    if (config.solverKind == SolverKind::gmg)
    {
        return std::make_unique<GmgStationarySolver>(
            exec, alpha, ux, lx, uy, ly, uz, lz, geom, bcArr, config
        );
    }
    return std::make_unique<FaceCoeffKrylovSolver>(
        exec, alpha, ux, lx, uy, ly, uz, lz, geom, bcArr, config
    );
}

} // namespace

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
    const SolverConfig& config
)
    : impl_(makeFaceCoeffSolver(makeExecutor(executor), alpha, ux, lx, uy, ly, uz, lz, geom, config)
    )
{}

SolveResult FaceCoeffSolver::solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
{
    return impl_->solve(rhs, sol);
}

namespace
{

// FaceCoeffCsrSolver has no matrix-free GMG hierarchy and no fp32 inner solve, so
// none of these 16 knobs do anything; the pre-refactor constructor accepted and
// silently discarded them (persistent.hpp:266-281, commented-out parameter names).
// Refused rather than ignored, for the same reason every other capability gap on
// this path is (see bc.hpp's checkBcData, gmg_bottom_solver in the FaceCoeffSolver
// ctor): accepting a knob that does nothing reports a configuration that was never
// applied.
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
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
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
    // The CSR assembly wraps neighbour indices around the domain
    // (periodic-only); parseBc also rejects a non-periodic geometry.
    const BcArray bcArr = parseBc(config.bc, geom, "FaceCoeffCsrSolver");
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
    if (config.bcData != nullptr)
    {
        throw std::runtime_error(
            "FaceCoeffCsrSolver: periodic boundaries only — bc_data needs FaceCoeffSolver"
        );
    }
    // config.precondKind's spelling was already validated once, by
    // parseSolverConfig; what is checked here is CSR's own, narrower
    // combination legality -- it has no matrix-free operator, so 'gmg' and
    // 'gmg_kokkos' (both otherwise-valid PrecondKind values) are refused, the
    // former with its own message.
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
    auto op = assembleFaceCoeffCsr(exec_, geom, *alpha, *ux, *lx, *uy, *ly, *uz, *lz);
    std::shared_ptr<const gko::LinOp> pc;
    if (config.precondMlmg != nullptr)
    {
        pc = gko::share(MlmgPrecond::create(
            exec_,
            config.precondMlmg,
            alpha->boxArray(),
            alpha->DistributionMap(),
            n_,
            config.precondCycles
        ));
    }
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

} // namespace blockamr::solvers
