// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/linearAlgebra/solve/oneshot.hpp"

#include <AMReX_Arena.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_MultiFabUtil.H>

#include <ginkgo/ginkgo.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"
#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/gkoTypes.hpp"
#include "NeoN/blockAmr/core/types.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/krylov.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/logging.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/mlmgOps.hpp"

namespace blockamr::la
{

using namespace amrex;

namespace
{

// gather/scatter walk host-side, so every scratch field here lives in pinned memory.
MultiFab pinnedFab(const BoxArray& ba, const DistributionMapping& dm, int nghost)
{
    return MultiFab(ba, dm, 1, nghost, MFInfo().SetArena(The_Pinned_Arena()));
}

// rtol is quoted against the ORIGINAL system's ||rhs||, but is applied as an ABSOLUTE
// criterion: relative to a correction system's own rhs a warm start would grind to reduce
// an already-converged residual again.
double absoluteStopTol(double rhsNorm, double rtol)
{
    return (rhsNorm > 0.0) ? rtol * rhsNorm : rtol;
}

// Explicit ||b - A x||_2 for reporting, independent of whatever the stopping criterion saw.
double finalResidualNorm(gko::LinOp& op, const Dense* b, const Dense* x)
{
    auto exec = b->get_executor();
    auto res = b->clone();
    auto one = gko::initialize<Dense>({1.0}, exec);
    auto negOne = gko::initialize<Dense>({-1.0}, exec);
    op.apply(negOne, x, one, res);
    auto norm = Dense::create(exec, gko::dim<2> {1, 1});
    res->compute_norm2(norm);
    return gko::clone(exec->get_master(), norm)->at(0, 0);
}

// r0 = rhs - L_inhom(x0), x0 = the incoming sol. MLMG::apply needs a ghost cell on the input
// and overwrites it, so sol's valid region goes into zero-initialized scratch.
MultiFab initialResidual(MLMG& mlmg, const MultiFab& sol, const MultiFab& rhs)
{
    const BoxArray& ba = sol.boxArray();
    const DistributionMapping& dm = sol.DistributionMap();
    MultiFab scratch = pinnedFab(ba, dm, 1);
    scratch.setVal(0.0);
    MultiFab::Copy(scratch, sol, 0, 0, 1, 0);
    MultiFab r0 = pinnedFab(ba, dm, 0);
    mlmg.apply({&r0}, {&scratch});
    // Xpay: dst = src + a*dst, i.e. r0 = rhs - L_inhom(x0).
    MultiFab::Xpay(r0, -1.0, rhs, 0, 0, 1, 0);
    return r0;
}

// Where each AMR level's block sits in the composite vector, coarsest first.
struct CompositeLayout
{
    // One MeshLevel per AMR level rather than a ba vector beside a dm vector: the two could
    // drift apart, and CompositeAmrexOp wants exactly this.
    Vector<MeshLevel> levels;
    Vector<long> offset;
    gko::size_type size;
    int numLevels;
};

CompositeLayout
makeCompositeLayout(const MLLinOpT<MultiFab>& lp, const Vector<MultiFab*>& sol, int nlevs)
{
    CompositeLayout layout;
    long ntot = 0;
    for (int lev = 0; lev < nlevs; ++lev)
    {
        layout.levels.push_back(
            MeshLevel {sol[lev]->boxArray(), sol[lev]->DistributionMap(), lp.Geom(lev)}
        );
        layout.offset.push_back(ntot);
        ntot += layout.levels.back().ba.numPts();
    }
    layout.size = static_cast<gko::size_type>(ntot);
    layout.numLevels = nlevs;
    return layout;
}

Vector<MultiFab> makeLevelFabs(const CompositeLayout& layout, int nghost)
{
    Vector<MultiFab> mfs(layout.numLevels);
    for (int lev = 0; lev < layout.numLevels; ++lev)
    {
        mfs[lev] = pinnedFab(layout.levels[lev].ba, layout.levels[lev].dm, nghost);
    }
    return mfs;
}

// Refinement ratio between AMR levels lev and lev+1, from the level domains
// (MLLinOp::AMRRefRatio is protected here).
IntVect refRatioToFiner(const MLLinOpT<MultiFab>& lp, int lev)
{
    const Box& cd = lp.Geom(lev).Domain();
    const Box& fd = lp.Geom(lev + 1).Domain();
    return IntVect(
        fd.length(0) / cd.length(0), fd.length(1) / cd.length(1), fd.length(2) / cd.length(2)
    );
}

// Covered coarse cells are slaved (zero operator columns), so their rhs entries must be the
// average_down of the fine rhs. Pinned copies; caller's rhs untouched.
Vector<MultiFab> makeConsistentRhs(
    const Vector<MultiFab const*>& rhs, const CompositeLayout& layout, const MLLinOpT<MultiFab>& lp
)
{
    Vector<MultiFab> rhsC = makeLevelFabs(layout, 0);
    for (int lev = 0; lev < layout.numLevels; ++lev)
    {
        MultiFab::Copy(rhsC[lev], *rhs[lev], 0, 0, 1, 0);
    }
    for (int lev = layout.numLevels - 2; lev >= 0; --lev)
    {
        average_down(rhsC[lev + 1], rhsC[lev], 0, 1, refRatioToFiner(lp, lev));
    }
    return rhsC;
}

// Per-level twin of initialResidual, against the consistency-corrected rhs.
Vector<MultiFab> initialLevelResiduals(
    MLMG& mlmg,
    const Vector<MultiFab*>& sol,
    const Vector<MultiFab>& rhsC,
    const CompositeLayout& layout
)
{
    Vector<MultiFab> scratch = makeLevelFabs(layout, 1);
    Vector<MultiFab> r0 = makeLevelFabs(layout, 0);
    Vector<MultiFab*> scratchP(layout.numLevels), r0P(layout.numLevels);
    for (int lev = 0; lev < layout.numLevels; ++lev)
    {
        scratch[lev].setVal(0.0);
        MultiFab::Copy(scratch[lev], *sol[lev], 0, 0, 1, 0);
        scratchP[lev] = &scratch[lev];
        r0P[lev] = &r0[lev];
    }
    mlmg.apply(r0P, scratchP);
    for (int lev = 0; lev < layout.numLevels; ++lev)
    {
        // Xpay: dst = src + a*dst, i.e. r0 = rhsC - L_inhom(x0).
        MultiFab::Xpay(r0[lev], -1.0, rhsC[lev], 0, 0, 1, 0);
    }
    return r0;
}

// L2 norm of the composite vector formed by stacking the per-level valid regions.
double compositeNorm2(const Vector<MultiFab>& mfs)
{
    double sumSq = 0.0;
    for (const MultiFab& mf : mfs)
    {
        const double nl = mf.norm2(0);
        sumSq += nl * nl;
    }
    return std::sqrt(sumSq);
}

// b = sign*r0 packed level-by-level. gather writes host-side, so b is built on the host
// master and then moved to exec.
std::unique_ptr<Dense> packLevels(
    std::shared_ptr<const gko::Executor> exec,
    const Vector<MultiFab>& r0,
    const CompositeLayout& layout,
    double sign
)
{
    auto bHost = Dense::create(exec->get_master(), gko::dim<2> {layout.size, 1});
    for (int lev = 0; lev < layout.numLevels; ++lev)
    {
        gather(r0[lev], bHost->get_values() + layout.offset[lev], sign);
    }
    return gko::clone(exec, bHost);
}

// sol = x0 + delta per level, then the covered-cell convention: covered coarse cells are the
// average_down of the fine solution, since the composite vector's covered entries are not DOFs.
void writeBackSolution(
    const double* x,
    const Vector<MultiFab*>& sol,
    const CompositeLayout& layout,
    const MLLinOpT<MultiFab>& lp
)
{
    for (int lev = 0; lev < layout.numLevels; ++lev)
    {
        MultiFab delta = pinnedFab(layout.levels[lev].ba, layout.levels[lev].dm, 0);
        scatter(x + layout.offset[lev], delta);
        MultiFab::Add(*sol[lev], delta, 0, 0, 1, 0);
    }
    for (int lev = layout.numLevels - 2; lev >= 0; --lev)
    {
        average_down(*sol[lev + 1], *sol[lev], 0, 1, refRatioToFiner(lp, lev));
    }
}

} // namespace

SolveResult
solveMlmgSystem(MLLinOpT<MultiFab>& lp, MultiFab& sol, const MultiFab& rhs, const OneshotSpec& spec)
{
    MLMG mlmg(lp);

    // SerialExecutor keeps the Krylov vector ops on the CPU, GPUExecutor on the device; the
    // mat-vec is on the GPU either way. The default is resolved at CALL time, not at binding
    // registration -- converting one there needs _neon's registrations, else bad_cast on import.
    auto exec = makeExecutor(spec.executor.value_or(NeoN::createDefaultExecutor()));
    const MeshLevel mesh {sol.boxArray(), sol.DistributionMap(), lp.Geom(0)};
    const auto n = static_cast<gko::size_type>(mesh.ba.numPts());

    // Op construction runs one apply to record c0 = L_inhom(0).
    auto op = gko::share(AmrexOp::create(exec, &mlmg, mesh, n, spec.sign));

    const MultiFab r0 = initialResidual(mlmg, sol, rhs);

    // b = sign*r0, matching the sign inside AmrexOp; the correction delta starts at zero.
    // gather writes host-side, so b is built on the host master and then moved to exec.
    auto bHost = Dense::create(exec->get_master(), gko::dim<2> {n, 1});
    gather(r0, bHost->get_values(), spec.sign);
    auto b = gko::clone(exec, bHost);
    auto x = Dense::create(exec, gko::dim<2> {n, 1});
    x->fill(0.0);

    // atol > 0 adds the plain stop ||r_k||_2 <= atol.
    const double stopTol = absoluteStopTol(rhs.norm2(0), spec.rtol);
    auto criteria =
        makeCriteria(exec, StopSpec {spec.maxIter, gko::stop::mode::absolute, stopTol, spec.atol});
    auto logger = gko::share(gko::log::Convergence<double>::create());
    auto resLogger = std::make_shared<ResidualHistoryLogger>();
    auto solver = gko::solver::Cg<double>::build().with_criteria(criteria).on(exec)->generate(op);
    solver->add_logger(logger);
    solver->add_logger(resLogger);
    solver->apply(b, x);

    // sol = x0 + delta.
    MultiFab delta = pinnedFab(mesh.ba, mesh.dm, 0);
    auto xHost = gko::clone(exec->get_master(), x);
    scatter(xHost->get_const_values(), delta);
    MultiFab::Add(sol, delta, 0, 0, 1, 0);

    return makeSolveResult(*logger, *resLogger, finalResidualNorm(*op, b.get(), x.get()));
}

SolveResult solveComposite(
    MLLinOpT<MultiFab>& lp,
    const Vector<MultiFab*>& sol,
    const Vector<MultiFab const*>& rhs,
    const OneshotSpec& spec
)
{
    const int nlevs = lp.NAMRLevels();

    MLMG mlmg(lp);

    auto exec = makeExecutor(spec.executor.value_or(NeoN::createDefaultExecutor()));

    const CompositeLayout layout = makeCompositeLayout(lp, sol, nlevs);

    // Op construction runs one apply to record c0 = L_inhom(0).
    auto op =
        gko::share(CompositeAmrexOp::create(exec, &mlmg, layout.levels, layout.size, spec.sign));

    const Vector<MultiFab> rhsC = makeConsistentRhs(rhs, layout, lp);
    const Vector<MultiFab> r0 = initialLevelResiduals(mlmg, sol, rhsC, layout);

    // The correction delta starts at zero.
    auto b = packLevels(exec, r0, layout, spec.sign);
    auto x = Dense::create(exec, gko::dim<2> {layout.size, 1});
    x->fill(0.0);

    // Stop on the composite ||rhs||_2 of the ORIGINAL system; atol > 0 adds ||r_k||_2 <= atol.
    const double stopTol = absoluteStopTol(compositeNorm2(rhsC), spec.rtol);
    auto criteria =
        makeCriteria(exec, StopSpec {spec.maxIter, gko::stop::mode::absolute, stopTol, spec.atol});
    auto logger = gko::share(gko::log::Convergence<double>::create());
    auto resLogger = std::make_shared<ResidualHistoryLogger>();
    std::shared_ptr<gko::LinOp> gsolver =
        generateBasicSolver(spec.solver, exec, op, criteria, "ginkgo_solve_composite");
    gsolver->add_logger(logger);
    gsolver->add_logger(resLogger);
    gsolver->apply(b, x);

    auto xHost = gko::clone(exec->get_master(), x);
    writeBackSolution(xHost->get_const_values(), sol, layout, lp);
    amrex::Gpu::streamSynchronize();

    return makeSolveResult(*logger, *resLogger, finalResidualNorm(*op, b.get(), x.get()));
}

SolveResult solveFaceCoeffs(
    const FaceCoeffLevel& level, MultiFab& sol, const MultiFab& rhs, const OneshotSpec& spec
)
{
    auto exec = gko::ReferenceExecutor::create();
    const auto n = static_cast<gko::size_type>(level.mesh.ba.numPts());

    // No NeoN executor reaches this entry point, and the Ginkgo one is fixed to Reference.
    auto op =
        gko::share(FaceCoeffOp::create(exec, NeoN::Executor {NeoN::SerialExecutor {}}, level));

    // Plain linear solve A x = b: the face coefficients are the full BC-folded matrix, so there
    // is no affine offset. Incoming sol seeds the initial guess, rhs is b.
    auto b = Dense::create(exec, gko::dim<2> {n, 1});
    gather(rhs, b->get_values(), 1.0);
    auto x = Dense::create(exec, gko::dim<2> {n, 1});
    gather(sol, x->get_values(), 1.0);

    const double stopTol = absoluteStopTol(rhs.norm2(0), spec.rtol);

    // spec.atol is 0.0 on this path -- `ginkgo_solve_face_coeffs` has no atol argument -- so
    // makeCriteria adds no absolute stop, exactly as the literal 0.0 here used to.
    auto criteria =
        makeCriteria(exec, StopSpec {spec.maxIter, gko::stop::mode::absolute, stopTol, spec.atol});

    auto logger = gko::share(gko::log::Convergence<double>::create());
    std::shared_ptr<gko::LinOp> gsolver =
        generateBasicSolver(spec.solver, exec, op, criteria, "ginkgo_solve_face_coeffs");
    gsolver->add_logger(logger);
    gsolver->apply(b, x);

    scatter(x->get_const_values(), sol);

    // Historical 2-key surface (num_iters, res_norm): the other fields are left unset, and
    // toDict omits keys for unset fields, so the Python-visible dict is unchanged.
    SolveResult result;
    result.num_iters = static_cast<std::int64_t>(logger->get_num_iterations());
    result.res_norm = finalResidualNorm(*op, b.get(), x.get());
    return result;
}

} // namespace blockamr::la
