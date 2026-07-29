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
#include "NeoN/blockAmr/core/types.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/krylov.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/logging.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/mlmgOps.hpp"

namespace blockamr::la
{

using namespace amrex;

SolveResult solveMlmgSystem(
    MLLinOpT<MultiFab>& lp,
    MultiFab& sol,
    const MultiFab& rhs,
    int max_iter,
    double rtol,
    double atol,
    double sign,
    std::optional<NeoN::Executor> executor
)
{
    MLMG mlmg(lp);

    // A SerialExecutor keeps the Krylov vector ops on the CPU; a
    // GPUExecutor runs them on the device. The mat-vec (MLMG::apply) is
    // on the GPU either way. The default is resolved at CALL time, not at
    // binding-registration time, so importing blockamr does not require
    // neon to have been imported first (converting a NeoN::Executor default
    // needs _neon's nb::class_ registrations, and getting that order wrong
    // raises std::bad_cast at import).
    auto exec = makeExecutor(executor.value_or(NeoN::createDefaultExecutor()));
    const BoxArray& ba = sol.boxArray();
    const DistributionMapping& dm = sol.DistributionMap();
    const auto n = static_cast<gko::size_type>(ba.numPts());

    // Op construction runs one apply to record c0 = L_inhom(0).
    auto op = gko::share(AmrexOp::create(exec, &mlmg, ba, dm, n, sign));

    // r0 = rhs - L_inhom(x0), x0 = incoming sol. MLMG::apply needs a
    // ghost cell on the input (and overwrites it), so copy sol's valid
    // region into a zero-initialized scratch rather than passing sol.
    MultiFab scratch(ba, dm, 1, 1, MFInfo().SetArena(The_Pinned_Arena()));
    scratch.setVal(0.0);
    MultiFab::Copy(scratch, sol, 0, 0, 1, 0);
    MultiFab r0(ba, dm, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));
    mlmg.apply({&r0}, {&scratch});
    // Xpay: dst = src + a*dst, i.e. r0 = rhs - L_inhom(x0).
    MultiFab::Xpay(r0, -1.0, rhs, 0, 0, 1, 0);

    // b = sign*r0, matching the sign inside AmrexOp; the correction
    // delta starts at zero.
    // gather writes host-side; build b on the executor's host master,
    // then move it to the (possibly device) solver executor.
    auto bHost = Dense::create(exec->get_master(), gko::dim<2> {n, 1});
    gather(r0, bHost->get_values(), sign);
    auto b = gko::clone(exec, bHost);
    auto x = Dense::create(exec, gko::dim<2> {n, 1});
    x->fill(0.0);

    // Stop on ||r_k|| <= rtol * ||rhs|| of the ORIGINAL system (an
    // absolute criterion here): the correction system's own rhs is
    // sign*r0, and relative to that a warm start (tiny r0) would grind
    // to reduce an already-converged residual by another factor rtol.
    // The correction residual equals the original-system residual, so
    // atol > 0 adds the plain absolute stop ||r_k|| <= atol.
    const double rhsNorm = rhs.norm2(0);
    const double stopTol = (rhsNorm > 0.0) ? rtol * rhsNorm : rtol;
    auto criteria = makeCriteria(exec, max_iter, gko::stop::mode::absolute, stopTol, atol);
    auto logger = gko::share(gko::log::Convergence<double>::create());
    auto resLogger = std::make_shared<ResidualHistoryLogger>();
    auto solver = gko::solver::Cg<double>::build().with_criteria(criteria).on(exec)->generate(op);
    solver->add_logger(logger);
    solver->add_logger(resLogger);
    solver->apply(b, x);

    // sol = x0 + delta.
    MultiFab delta(ba, dm, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));
    auto xHost = gko::clone(exec->get_master(), x);
    scatter(xHost->get_const_values(), delta);
    MultiFab::Add(sol, delta, 0, 0, 1, 0);

    // Explicit final residual ||b - A_home delta||_2 for reporting.
    auto res = b->clone();
    auto one = gko::initialize<Dense>({1.0}, exec);
    auto negOne = gko::initialize<Dense>({-1.0}, exec);
    op->apply(negOne, x, one, res);
    auto norm = Dense::create(exec, gko::dim<2> {1, 1});
    res->compute_norm2(norm);
    auto normHost = gko::clone(exec->get_master(), norm);

    return makeSolveResult(*logger, *resLogger, normHost->at(0, 0));
}

SolveResult solveComposite(
    MLLinOpT<MultiFab>& lp,
    const Vector<MultiFab*>& sol,
    const Vector<MultiFab const*>& rhs,
    int max_iter,
    double rtol,
    double atol,
    double sign,
    std::optional<NeoN::Executor> executor,
    const std::string& solver
)
{
    const int nlevs = lp.NAMRLevels();

    MLMG mlmg(lp);

    auto exec = makeExecutor(executor.value_or(NeoN::createDefaultExecutor()));

    std::vector<BoxArray> bas;
    std::vector<DistributionMapping> dms;
    std::vector<long> off;
    long ntot = 0;
    for (int lev = 0; lev < nlevs; ++lev)
    {
        bas.push_back(sol[static_cast<std::size_t>(lev)]->boxArray());
        dms.push_back(sol[static_cast<std::size_t>(lev)]->DistributionMap());
        off.push_back(ntot);
        ntot += bas.back().numPts();
    }
    const auto n = static_cast<gko::size_type>(ntot);

    // Op construction runs one apply to record c0 = L_inhom(0).
    auto op = gko::share(CompositeAmrexOp::create(exec, &mlmg, bas, dms, n, sign));

    // Refinement ratio between AMR levels lev and lev+1, from the
    // level domains (MLLinOp::AMRRefRatio is protected here).
    auto refRatio = [&lp](int lev)
    {
        const Box& cd = lp.Geom(lev).Domain();
        const Box& fd = lp.Geom(lev + 1).Domain();
        return IntVect(
            fd.length(0) / cd.length(0), fd.length(1) / cd.length(1), fd.length(2) / cd.length(2)
        );
    };

    // Consistent rhs: coarse cells covered by a finer level are slaved
    // (their operator columns are zero — see CompositeAmrexOp), so
    // their rhs entries must be the average_down of the fine rhs for
    // the system to be solvable. Pinned copies; caller's rhs untouched.
    Vector<MultiFab> rhsC(nlevs);
    for (int lev = 0; lev < nlevs; ++lev)
    {
        rhsC[lev].define(
            bas[static_cast<std::size_t>(lev)],
            dms[static_cast<std::size_t>(lev)],
            1,
            0,
            MFInfo().SetArena(The_Pinned_Arena())
        );
        MultiFab::Copy(rhsC[lev], *rhs[static_cast<std::size_t>(lev)], 0, 0, 1, 0);
    }
    for (int lev = nlevs - 2; lev >= 0; --lev)
    {
        average_down(rhsC[lev + 1], rhsC[lev], 0, 1, refRatio(lev));
    }

    // r0 = rhs - L_inhom(x0), x0 = incoming sol (per level). MLMG::apply
    // needs a ghost cell on the input (and overwrites it), so copy sol's
    // valid region into zero-initialized scratch rather than passing sol.
    Vector<MultiFab> scratch(nlevs), r0(nlevs);
    Vector<MultiFab*> scratchP(nlevs), r0P(nlevs);
    for (int lev = 0; lev < nlevs; ++lev)
    {
        scratch[lev].define(
            bas[static_cast<std::size_t>(lev)],
            dms[static_cast<std::size_t>(lev)],
            1,
            1,
            MFInfo().SetArena(The_Pinned_Arena())
        );
        scratch[lev].setVal(0.0);
        MultiFab::Copy(scratch[lev], *sol[static_cast<std::size_t>(lev)], 0, 0, 1, 0);
        r0[lev].define(
            bas[static_cast<std::size_t>(lev)],
            dms[static_cast<std::size_t>(lev)],
            1,
            0,
            MFInfo().SetArena(The_Pinned_Arena())
        );
        scratchP[lev] = &scratch[lev];
        r0P[lev] = &r0[lev];
    }
    mlmg.apply(r0P, scratchP);
    for (int lev = 0; lev < nlevs; ++lev)
    {
        // Xpay: dst = src + a*dst, i.e. r0 = rhsC - L_inhom(x0).
        MultiFab::Xpay(r0[lev], -1.0, rhsC[lev], 0, 0, 1, 0);
    }

    // b = sign*r0 packed level-by-level; the correction delta starts
    // at zero.
    auto bHost = Dense::create(exec->get_master(), gko::dim<2> {n, 1});
    for (int lev = 0; lev < nlevs; ++lev)
    {
        gather(r0[lev], bHost->get_values() + off[static_cast<std::size_t>(lev)], sign);
    }
    auto b = gko::clone(exec, bHost);
    auto x = Dense::create(exec, gko::dim<2> {n, 1});
    x->fill(0.0);

    // Stop on the composite ||rhs|| of the ORIGINAL system, as an
    // absolute criterion (see solveMlmgSystem for the warm-start rationale).
    double rhsNorm2 = 0.0;
    for (int lev = 0; lev < nlevs; ++lev)
    {
        const double nl = rhsC[lev].norm2(0);
        rhsNorm2 += nl * nl;
    }
    const double rhsNorm = std::sqrt(rhsNorm2);
    const double stopTol = (rhsNorm > 0.0) ? rtol * rhsNorm : rtol;
    auto criteria = makeCriteria(exec, max_iter, gko::stop::mode::absolute, stopTol, atol);
    auto logger = gko::share(gko::log::Convergence<double>::create());
    auto resLogger = std::make_shared<ResidualHistoryLogger>();
    std::shared_ptr<gko::LinOp> gsolver =
        generateBasicSolver(solver, exec, op, criteria, "ginkgo_solve_composite");
    gsolver->add_logger(logger);
    gsolver->add_logger(resLogger);
    gsolver->apply(b, x);

    // sol = x0 + delta per level, then enforce the covered-cell
    // convention: coarse covered cells = average_down of the fine
    // solution (matching MLMG::solve — the covered entries of x are
    // Krylov by-products, not DOFs).
    auto xHost = gko::clone(exec->get_master(), x);
    for (int lev = 0; lev < nlevs; ++lev)
    {
        MultiFab delta(
            bas[static_cast<std::size_t>(lev)],
            dms[static_cast<std::size_t>(lev)],
            1,
            0,
            MFInfo().SetArena(The_Pinned_Arena())
        );
        scatter(xHost->get_const_values() + off[static_cast<std::size_t>(lev)], delta);
        MultiFab::Add(*sol[static_cast<std::size_t>(lev)], delta, 0, 0, 1, 0);
    }
    for (int lev = nlevs - 2; lev >= 0; --lev)
    {
        average_down(
            *sol[static_cast<std::size_t>(lev + 1)],
            *sol[static_cast<std::size_t>(lev)],
            0,
            1,
            refRatio(lev)
        );
    }
    amrex::Gpu::streamSynchronize();

    // Explicit final residual ||b - A_home delta||_2 for reporting.
    auto res = b->clone();
    auto one = gko::initialize<Dense>({1.0}, exec);
    auto negOne = gko::initialize<Dense>({-1.0}, exec);
    op->apply(negOne, x, one, res);
    auto norm = Dense::create(exec, gko::dim<2> {1, 1});
    res->compute_norm2(norm);
    auto normHost = gko::clone(exec->get_master(), norm);

    return makeSolveResult(*logger, *resLogger, normHost->at(0, 0));
}

SolveResult solveFaceCoeffs(
    MultiFab& alpha,
    MultiFab& ux,
    MultiFab& lx,
    MultiFab& uy,
    MultiFab& ly,
    MultiFab& uz,
    MultiFab& lz,
    MultiFab& sol,
    const MultiFab& rhs,
    const Geometry& geom,
    const std::string& solver,
    int max_iter,
    double rtol
)
{
    auto exec = gko::ReferenceExecutor::create();
    const BoxArray& ba = sol.boxArray();
    const DistributionMapping& dm = sol.DistributionMap();
    const auto n = static_cast<gko::size_type>(ba.numPts());

    // No NeoN executor reaches this entry point; the Ginkgo one above is fixed to
    // Reference, so the matching NeoN alternative is SerialExecutor.
    auto op = gko::share(FaceCoeffOp::create(
        exec,
        NeoN::Executor {NeoN::SerialExecutor {}},
        MeshLevel {ba, dm, geom},
        n,
        // Non-owning handles: this entry point is handed the caller's MultiFabs
        // by reference and does not take ownership of them.
        CellFieldLevel {nonOwning(alpha)},
        FaceFieldLevel {{nonOwning(ux), nonOwning(uy), nonOwning(uz)}},
        FaceFieldLevel {{nonOwning(lx), nonOwning(ly), nonOwning(lz)}}
    ));

    // Plain linear solve A x = b: the face coefficients are the full
    // (BC-folded) matrix, so no affine offset. Incoming sol seeds the
    // initial guess (Ginkgo uses x's initial values), rhs is b.
    auto b = Dense::create(exec, gko::dim<2> {n, 1});
    gather(rhs, b->get_values(), 1.0);
    auto x = Dense::create(exec, gko::dim<2> {n, 1});
    gather(sol, x->get_values(), 1.0);

    const double rhsNorm = rhs.norm2(0);
    const double stopTol = (rhsNorm > 0.0) ? rtol * rhsNorm : rtol;

    auto criteria = makeCriteria(exec, max_iter, gko::stop::mode::absolute, stopTol, 0.0);

    auto logger = gko::share(gko::log::Convergence<double>::create());
    std::shared_ptr<gko::LinOp> gsolver =
        generateBasicSolver(solver, exec, op, criteria, "ginkgo_solve_face_coeffs");
    gsolver->add_logger(logger);
    gsolver->apply(b, x);

    scatter(x->get_const_values(), sol);

    // Explicit final residual ||b - A x||_2 for reporting.
    auto res = b->clone();
    auto one = gko::initialize<Dense>({1.0}, exec);
    auto negOne = gko::initialize<Dense>({-1.0}, exec);
    op->apply(negOne, x, one, res);
    auto norm = Dense::create(exec, gko::dim<2> {1, 1});
    res->compute_norm2(norm);

    // Historical 2-key surface only (num_iters, res_norm): converged/
    // res_history/contraction/diagnostic are left unset, and the nanobind
    // converter in ginkgoSolve.cpp omits keys for unset fields, so the
    // Python-visible dict is unchanged.
    SolveResult result;
    result.num_iters = static_cast<std::int64_t>(logger->get_num_iterations());
    result.res_norm = norm->at(0, 0);
    return result;
}

} // namespace blockamr::la
