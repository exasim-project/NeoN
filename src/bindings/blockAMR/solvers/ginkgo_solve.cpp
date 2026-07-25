// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Matrix-free Ginkgo CG solve of an AMReX MLLinOp system (single-level, CPU
// serial). The mat-vec is MLMG::apply, which computes out = L_inhom(in): the
// operator evaluated with the inhomogeneous BC data set via set_level_bc, so
// it is AFFINE, not linear. The solve therefore runs in residual-correction
// form: with x0 the incoming sol and c0 = L_inhom(0) the constant offset,
//   A_home(v) = sign * (L_inhom(v) - c0)          (linear)
//   A_home(delta) = sign * (rhs - L_inhom(x0)),   sol = x0 + delta.
// `sign` makes the operator SPD for CG: -1 for MLPoisson (L = +laplacian,
// negative-definite), +1 for MLABecLaplacian (alpha*a*phi - beta*div(b
// grad phi), already positive-definite). With homogeneous BCs and x0 = 0,
// c0 = 0 and r0 = rhs, so this reduces exactly to a plain CG solve of
// sign*L (the milestone-1 behavior).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <AMReX_Arena.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>

#include <ginkgo/ginkgo.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../bindings.hpp"

#include "face_coeff_op.hpp"
#include "krylov.hpp"
#include "mlmg_ops.hpp"
#include "persistent.hpp"
#include "profiling.hpp"
#include "transfer.hpp"
#include "types.hpp"

namespace nb = nanobind;

using namespace blockamr::solvers;

namespace
{

// Bind a persistent solver class S (constructor: coefficients + geom + config;
// method: solve(rhs, sol)). keep_alive ties the coefficient fields to the
// solver, since the matrix-free operator references them on the device.
template<class S>
void bindPersistent(nb::module_& m, const char* name)
{
    nb::class_<S>(m, name)
        .def(
            "__init__",
            [](S* self,
               amrex::MultiFab& alpha,
               amrex::MultiFab& ux,
               amrex::MultiFab& lx,
               amrex::MultiFab& uy,
               amrex::MultiFab& ly,
               amrex::MultiFab& uz,
               amrex::MultiFab& lz,
               const amrex::Geometry& geom,
               const std::string& executor,
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
               double gmg_omega)
            {
                new (self)
                    S(executor,
                      geom,
                      &alpha,
                      &ux,
                      &lx,
                      &uy,
                      &ly,
                      &uz,
                      &lz,
                      solver,
                      max_iter,
                      rtol,
                      atol,
                      project_nullspace,
                      precond_mlmg,
                      precond_cycles,
                      bc,
                      precond,
                      gmg_pre_sweeps,
                      gmg_post_sweeps,
                      gmg_coarsest_sweeps,
                      gmg_max_levels,
                      gmg_min_bottom,
                      gmg_smoother,
                      gmg_precision,
                      gmg_omega);
            },
            nb::arg("alpha"),
            nb::arg("ux"),
            nb::arg("lx"),
            nb::arg("uy"),
            nb::arg("ly"),
            nb::arg("uz"),
            nb::arg("lz"),
            nb::arg("geom"),
            nb::arg("executor") = "cuda",
            // Krylov solvers "cg" | "bicgstab" | "gmres", OR "gmg" (matrix-free
            // solver only): the NATIVE stationary geometric-multigrid solver
            // x <- x + V(b - A x) run to tolerance (Richardson iteration, like
            // MLMG) — no Ginkgo Krylov object, the whole loop on AMReX fabs.
            // solver="gmg" builds the V-cycle hierarchy directly and IGNORES the
            // `precond` argument (the V-cycle IS the solver). A standalone
            // V-cycle needs the coarsest grid solved accurately, so raise
            // gmg_coarsest_sweeps (~100 for rbgs, ~160 for chebyshev) — the
            // CG-tuned default of 8 gives a weak, slowly-converging iteration.
            // solver="ir" is the Ginkgo-idiomatic twin of "gmg": a
            // gko::solver::Ir<double> (iterative refinement, relaxation 1.0) whose
            // system matrix is the matrix-free FaceCoeffOp and whose inner solver is
            // the generated GMG V-cycle LinOp. Same GMG semantics (builds the
            // hierarchy, ignores `precond`, needs the accurate coarsest solve) but
            // driven through Ginkgo's Dense pack/unpack + Convergence logger.
            nb::arg("solver") = "bicgstab",
            nb::arg("max_iter") = 1000,
            nb::arg("rtol") = 1e-10,
            nb::arg("atol") = 0.0,
            nb::arg("project_nullspace") = false,
            nb::arg("precond_mlmg").none() = nb::none(),
            nb::arg("precond_cycles") = 1,
            // Domain BCs, order (xlo, xhi, ylo, yhi, zlo, zhi); each entry is
            // "periodic", "dirichlet" (homogeneous, u=0 on the face) or
            // "neumann" (homogeneous, du/dn=0). Must match the geometry's
            // periodicity per direction. Matrix-free solver only.
            nb::arg("bc") = std::vector<std::string>(6, "periodic"),
            // Preconditioner selector: "none" (default; precond_mlmg alone
            // implies "mlmg"), "mlmg" (requires precond_mlmg) or "gmg" (native
            // matrix-free geometric multigrid on the face coefficients —
            // matrix-free solver only, no MLMG involved).
            nb::arg("precond") = "none",
            // Native-GMG (precond="gmg") V-cycle knobs. Defaults reproduce the
            // previous fixed behaviour. gmg_pre_sweeps/gmg_post_sweeps: RB-GS
            // sweep count / Chebyshev degree per pre-/post-smooth (keep them
            // equal for a CG-safe symmetric V-cycle). gmg_coarsest_sweeps:
            // smoothing on the bottom level. gmg_max_levels: 0 = auto/unlimited
            // coarsening; else cap the hierarchy depth. gmg_min_bottom: stop
            // coarsening before the domain shortside drops below this.
            // gmg_smoother: "rbgs" (red-black Gauss-Seidel) or "chebyshev"
            // (Jacobi-preconditioned polynomial, plain-stencil bandwidth).
            nb::arg("gmg_pre_sweeps") = 2,
            nb::arg("gmg_post_sweeps") = 2,
            nb::arg("gmg_coarsest_sweeps") = 8,
            nb::arg("gmg_max_levels") = 0,
            nb::arg("gmg_min_bottom") = 4,
            nb::arg("gmg_smoother") = "rbgs",
            // Native-GMG hierarchy precision: "fp64" (default; byte-for-byte the
            // previous behaviour) or "fp32" — the whole V-cycle (level
            // coefficients, work fields, smoother, restriction/prolongation,
            // ghost fills) runs in single precision while the outer CG/operator
            // stays double, halving the bandwidth-bound V-cycle traffic.
            // Matrix-free solver only.
            nb::arg("gmg_precision") = "fp64",
            // RB-SOR relaxation factor for gmg_smoother="rbgs":
            //   sol <- sol + gmg_omega * (gs - sol)
            // 1.0 (default) is plain red-black Gauss-Seidel, bit-for-bit the
            // previous behaviour. MLMG's own abec_gsrb over-relaxes with 1.15.
            // Must lie in (0, 2) for a convergent relaxation. Ignored by
            // gmg_smoother="chebyshev", whose damping comes from the polynomial.
            // NOTE: omega != 1.0 makes the colour sweep non-symmetric, so the
            // V-cycle is no longer exactly self-adjoint even with the reversed
            // post-smooth. That is harmless for solver="gmg"/"ir" (stationary
            // iterations), but can degrade CG, which assumes an SPD
            // preconditioner — prefer omega = 1.0 or "chebyshev" under precond="gmg".
            nb::arg("gmg_omega") = 1.0,
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>(),
            nb::keep_alive<1, 4>(),
            nb::keep_alive<1, 5>(),
            nb::keep_alive<1, 6>(),
            nb::keep_alive<1, 7>(),
            nb::keep_alive<1, 8>(),
            // The preconditioner MLMG (arg 16; self=1, args from 2) must
            // outlive the solver — MlmgPrecond holds a raw pointer to it.
            // keep_alive is a no-op when the arg is None.
            nb::keep_alive<1, 16>()
        )
        .def(
            "solve",
            [](S& self, amrex::MultiFab& rhs, amrex::MultiFab& sol)
            { return self.solve(rhs, sol); },
            nb::arg("rhs"),
            nb::arg("sol"),
            "Solve A sol = rhs, reusing the prebuilt operator and solver. sol's\n"
            "incoming values seed the initial guess; the matrix is defined by the\n"
            "coefficient fields handed to the constructor (and, for the matrix-free\n"
            "solver, re-read each call so in-place updates take effect). With\n"
            "project_nullspace=True (constructor kwarg, for singular systems with\n"
            "the constant nullspace, e.g. fully-periodic pure Poisson) the rhs and\n"
            "initial guess are projected mean-zero before the Krylov solve and the\n"
            "returned solution is the mean-zero representative. With precond_mlmg\n"
            "(constructor kwarg: an MLMG built on an equivalent operator) each\n"
            "Krylov iteration is preconditioned by precond_cycles multigrid\n"
            "V-cycles, keeping the iteration count ~flat in N. precond='gmg'\n"
            "(constructor kwarg, matrix-free solver only) instead uses the\n"
            "native geometric-multigrid V-cycle built directly on the face\n"
            "coefficients (no MLMG anywhere). bc (constructor\n"
            "kwarg, matrix-free solver only): 6 entries (xlo, xhi, ylo, yhi,\n"
            "zlo, zhi) of 'periodic' | 'dirichlet' | 'neumann' — homogeneous\n"
            "domain BCs folded in via ghost reflection; must match the\n"
            "geometry's periodicity per direction. Returns a\n"
            "dict with num_iters, res_norm, converged and res_history (per-\n"
            "iteration residual norms of this call)."
        );
}

} // namespace

void registerGinkgoSolve(nb::module_& m)
{
    using namespace amrex;

    using MLLinOp = MLLinOpT<MultiFab>;

    m.def(
        "ginkgo_solve",
        [](MLLinOp& lp,
           MultiFab& sol,
           const MultiFab& rhs,
           int max_iter,
           double rtol,
           double atol,
           double sign,
           const std::string& executor)
        {
            MLMG mlmg(lp);

            // "reference" keeps the Krylov vector ops on the CPU; "cuda" runs
            // them on the GPU (device 0) with a ReferenceExecutor as host
            // master. The mat-vec (MLMG::apply) is on the GPU either way.
            auto exec = makeExecutor(executor);
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
            auto solver =
                gko::solver::Cg<double>::build().with_criteria(criteria).on(exec)->generate(op);
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

            return makeResultDict(*logger, *resLogger, normHost->at(0, 0));
        },
        nb::arg("lp"),
        nb::arg("sol"),
        nb::arg("rhs"),
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        nb::arg("atol") = 0.0,
        nb::arg("sign") = -1.0,
        nb::arg("executor") = "reference",
        "Matrix-free Ginkgo CG solve of the MLLinOp system L(sol) = rhs.\n\n"
        "sol's incoming values are the initial guess, and boundary data set\n"
        "via set_level_bc is honored (residual-correction solve). `sign` must\n"
        "make sign*L SPD: -1.0 (default) for MLPoisson (L = +laplacian,\n"
        "negative-definite); +1.0 for MLABecLaplacian (alpha*a*phi -\n"
        "beta*div(b grad phi), positive-definite). CG stops when\n"
        "||r_k|| <= rtol*||rhs|| (or ||r_k|| <= atol when atol > 0), so a warm\n"
        "start converges immediately.\n"
        "`executor` is 'reference' (CPU, default) or 'cuda' (GPU device 0). On\n"
        "'cuda' the entire solve runs on the device: the Krylov vector ops, the\n"
        "MLMG::apply mat-vec, and the vector<->MultiFab pack/unpack kernels all\n"
        "stay on the GPU, with no per-iteration host transfer. Returns a dict\n"
        "with num_iters, res_norm (2-norm of the homogeneous-system residual),\n"
        "converged and res_history (per-iteration residual norms)."
    );

    m.def(
        "ginkgo_solve_composite",
        [](MLLinOp& lp,
           nb::list sol_py,
           nb::list rhs_py,
           int max_iter,
           double rtol,
           double atol,
           double sign,
           const std::string& executor,
           const std::string& solver)
        {
            const int nlevs = lp.NAMRLevels();
            if (static_cast<int>(nb::len(sol_py)) != nlevs
                || static_cast<int>(nb::len(rhs_py)) != nlevs)
            {
                throw std::runtime_error(
                    "ginkgo_solve_composite: sol and rhs need one MultiFab per AMR level ("
                    + std::to_string(nlevs) + ")"
                );
            }
            Vector<MultiFab*> sol(nlevs);
            Vector<MultiFab const*> rhs(nlevs);
            for (int lev = 0; lev < nlevs; ++lev)
            {
                sol[lev] = &nb::cast<MultiFab&>(sol_py[static_cast<std::size_t>(lev)]);
                rhs[lev] = &nb::cast<MultiFab const&>(rhs_py[static_cast<std::size_t>(lev)]);
            }

            MLMG mlmg(lp);

            auto exec = makeExecutor(executor);

            std::vector<BoxArray> bas;
            std::vector<DistributionMapping> dms;
            std::vector<long> off;
            long ntot = 0;
            for (int lev = 0; lev < nlevs; ++lev)
            {
                bas.push_back(sol[lev]->boxArray());
                dms.push_back(sol[lev]->DistributionMap());
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
                    fd.length(0) / cd.length(0),
                    fd.length(1) / cd.length(1),
                    fd.length(2) / cd.length(2)
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
                MultiFab::Copy(rhsC[lev], *rhs[lev], 0, 0, 1, 0);
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
                MultiFab::Copy(scratch[lev], *sol[lev], 0, 0, 1, 0);
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
            // absolute criterion (see ginkgo_solve for the warm-start rationale).
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
            std::shared_ptr<gko::LinOp> gsolver;
            if (solver == "cg")
            {
                gsolver =
                    gko::solver::Cg<double>::build().with_criteria(criteria).on(exec)->generate(op);
            }
            else if (solver == "bicgstab")
            {
                gsolver = gko::solver::Bicgstab<double>::build()
                              .with_criteria(criteria)
                              .on(exec)
                              ->generate(op);
            }
            else if (solver == "gmres")
            {
                gsolver =
                    gko::solver::Gmres<double>::build().with_criteria(criteria).on(exec)->generate(
                        op
                    );
            }
            else
            {
                throw std::runtime_error("ginkgo_solve_composite: unknown solver '" + solver + "'");
            }
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
                MultiFab::Add(*sol[lev], delta, 0, 0, 1, 0);
            }
            for (int lev = nlevs - 2; lev >= 0; --lev)
            {
                average_down(*sol[lev + 1], *sol[lev], 0, 1, refRatio(lev));
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

            return makeResultDict(*logger, *resLogger, normHost->at(0, 0));
        },
        nb::arg("lp"),
        nb::arg("sol"),
        nb::arg("rhs"),
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        nb::arg("atol") = 0.0,
        nb::arg("sign") = -1.0,
        nb::arg("executor") = "reference",
        nb::arg("solver") = "bicgstab",
        "Matrix-free Ginkgo solve of the multi-level COMPOSITE MLLinOp system\n"
        "L(sol) = rhs on a 2+ level AMR hierarchy (one sol/rhs MultiFab per\n"
        "level, coarsest first). The mat-vec is the multi-level MLMG::apply:\n"
        "coarse/fine interface interpolation, reflux and covered-cell\n"
        "average_down are all handled by AMReX, so the solved system is\n"
        "identical to MLMG's own composite solve. Covered coarse cells are\n"
        "slaved, not DOFs: their rhs entries are replaced internally by the\n"
        "average_down of the fine rhs, and on return they hold the\n"
        "average_down of the fine solution. sol's incoming values are the\n"
        "initial guess (residual-correction form, set_level_bc honored).\n"
        "`sign` as in ginkgo_solve: -1.0 for MLPoisson, +1.0 for\n"
        "MLABecLaplacian. The composite operator is not exactly symmetric\n"
        "(c/f interpolation vs reflux), so solver='bicgstab' (default) or\n"
        "'gmres' are safe; 'cg' may work in practice. Stops when\n"
        "||r_k|| <= rtol*||rhs|| (composite norm; or ||r_k|| <= atol when\n"
        "atol > 0). executor='reference'|'cuda'. Returns a dict with\n"
        "num_iters, res_norm, converged and res_history."
    );

    m.def(
        "ginkgo_solve_face_coeffs",
        [](MultiFab& alpha,
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
           double rtol)
        {
            auto exec = gko::ReferenceExecutor::create();
            const BoxArray& ba = sol.boxArray();
            const DistributionMapping& dm = sol.DistributionMap();
            const auto n = static_cast<gko::size_type>(ba.numPts());

            auto op = gko::share(
                FaceCoeffOp::create(exec, ba, dm, geom, n, &alpha, &ux, &lx, &uy, &ly, &uz, &lz)
            );

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
            std::shared_ptr<gko::LinOp> gsolver;
            if (solver == "cg")
            {
                gsolver =
                    gko::solver::Cg<double>::build().with_criteria(criteria).on(exec)->generate(op);
            }
            else if (solver == "bicgstab")
            {
                gsolver = gko::solver::Bicgstab<double>::build()
                              .with_criteria(criteria)
                              .on(exec)
                              ->generate(op);
            }
            else if (solver == "gmres")
            {
                gsolver =
                    gko::solver::Gmres<double>::build().with_criteria(criteria).on(exec)->generate(
                        op
                    );
            }
            else
            {
                throw std::runtime_error(
                    "ginkgo_solve_face_coeffs: unknown solver '" + solver + "'"
                );
            }
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

            nb::dict result;
            result["num_iters"] = static_cast<std::int64_t>(logger->get_num_iterations());
            result["res_norm"] = norm->at(0, 0);
            return result;
        },
        nb::arg("alpha"),
        nb::arg("ux"),
        nb::arg("lx"),
        nb::arg("uy"),
        nb::arg("ly"),
        nb::arg("uz"),
        nb::arg("lz"),
        nb::arg("sol"),
        nb::arg("rhs"),
        nb::arg("geom"),
        nb::arg("solver") = "bicgstab",
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        "Matrix-free Ginkgo solve of a general structured face-coefficient system A(sol) = rhs.\n\n"
        "The matrix is carried as OpenFOAM-style AMReX fields: alpha is the\n"
        "cell-centred diagonal SOURCE (ddt/Sp/reaction), and u{x,y,z}/l{x,y,z}\n"
        "are the face-centred upper/lower off-diagonal coefficients (pass the\n"
        "same field for u* and l* for a symmetric matrix). The full diagonal is\n"
        "assembled on the fly as alpha - negSumDiag(faces). `solver` is one of\n"
        "'cg' (SPD only), 'bicgstab' (default), or 'gmres'. sol's incoming\n"
        "values seed the initial guess. CG/BiCGStab/GMRES stop when\n"
        "||r_k|| <= rtol*||rhs||. Returns a dict with num_iters and res_norm."
    );

    // Persistent solvers: build the operator + Ginkgo solver once, solve many
    // times. FaceCoeffSolver is matrix-free (recomputes the mat-vec from the
    // face coefficients each apply); FaceCoeffCsrSolver assembles the same
    // matrix into a CSR (single-box periodic) so the benefit of matrix-free
    // over an explicit sparse matrix can be measured.
    bindPersistent<FaceCoeffSolver>(m, "FaceCoeffSolver");
    bindPersistent<FaceCoeffCsrSolver>(m, "FaceCoeffCsrSolver");

    // M0 profiling accessors (see namespace prof). Empty unless the process
    // runs with BLOCKAMR_PROFILE=1.
    m.def(
        "profile_report",
        []()
        {
            nb::dict d;
            for (const auto& [key, acc] : prof::table())
            {
                d[key.c_str()] = nb::make_tuple(acc.sec, acc.count);
            }
            return d;
        },
        "Accumulated {phase: (seconds, count)} timers (BLOCKAMR_PROFILE=1)."
    );
    m.def(
        "profile_reset",
        []() { prof::table().clear(); },
        "Clear the BLOCKAMR_PROFILE=1 phase-timer accumulators."
    );
}
