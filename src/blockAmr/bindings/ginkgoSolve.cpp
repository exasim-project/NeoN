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
// The caster for NeoN::Executor, which is a std::variant of the three executor
// classes bound in bindings/executors.cpp.
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>

#include <ginkgo/ginkgo.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "bindings.hpp"

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/core/profiling.hpp"
#include "NeoN/blockAmr/core/types.hpp"
#include "NeoN/blockAmr/linearAlgebra/faceCoeffMatrix.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/result.hpp"
#include "NeoN/blockAmr/linearAlgebra/linearSystem.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrix.hpp"
#include "NeoN/blockAmr/linearAlgebra/operator.hpp"
#include "NeoN/blockAmr/linearAlgebra/solve/oneshot.hpp"
#include "NeoN/blockAmr/linearAlgebra/solve/persistent.hpp"
#include "NeoN/blockAmr/linearAlgebra/solver.hpp"
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"
#include "NeoN/blockAmr/operators/laplacian.hpp"

namespace nb = nanobind;

using namespace blockamr::la;

namespace
{

// The one place a SolveResult crosses into Python. Every solve entry point in
// this file returns a SolveResult (the library stack is nanobind-free); this
// converts it to the dict Python callers see. converged/res_history/
// contraction/diagnostic are only present in the dict when the corresponding
// SolveResult field is set, so ginkgo_solve_face_coeffs' historical 2-key
// surface (num_iters, res_norm only) round-trips through here byte-for-byte.
nb::dict toDict(const SolveResult& r)
{
    nb::dict d;
    d["num_iters"] = r.num_iters;
    d["res_norm"] = r.res_norm;
    if (r.converged)
    {
        d["converged"] = *r.converged;
    }
    if (r.res_history)
    {
        nb::list hist;
        for (double v : *r.res_history)
        {
            hist.append(v);
        }
        d["res_history"] = hist;
    }
    if (r.contraction)
    {
        d["contraction"] = *r.contraction;
    }
    if (r.diagnostic)
    {
        d["diagnostic"] = *r.diagnostic;
    }
    return d;
}

// Repackages the 27 non-fixed __init__ arguments (36 nb::arg total minus the 9
// fixed: executor, geom, alpha..lz) into one SolverConfig, built once here so
// FaceCoeffSolver/FaceCoeffCsrSolver take `const SolverConfig&` instead of
// spelling out all 27. Zero nanobind types; also parses solver/precond to
// their enums (parseSolverKind/parsePrecondKind throw on unknown spellings).
SolverConfig parseSolverConfig(
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
{
    SolverConfig config;
    config.solver = solver;
    // Parsed once, here, rather than wherever a solver name/precond first
    // happens to be dispatched on: an unknown spelling is now rejected before
    // any other constructor work runs, with the same message either check
    // threw before (see solverConfig.hpp).
    config.solverKind = parseSolverKind(solver);
    config.maxIter = max_iter;
    config.rtol = rtol;
    config.atol = atol;
    config.projectNullspace = project_nullspace;
    config.precondMlmg = precond_mlmg;
    config.precondCycles = precond_cycles;
    config.bc = bc;
    config.precond = precond;
    config.precondKind = parsePrecondKind(precond);
    config.gmg.preSweeps = gmg_pre_sweeps;
    config.gmg.postSweeps = gmg_post_sweeps;
    config.gmg.coarsestSweeps = gmg_coarsest_sweeps;
    config.gmg.maxLevels = gmg_max_levels;
    config.gmg.minBottom = gmg_min_bottom;
    config.gmg.smoother = gmg_smoother;
    config.gmg.precision = gmg_precision;
    config.gmg.coeffPrecision = gmg_coeff_precision;
    config.gmg.omega = gmg_omega;
    config.gmg.aggLevel0Size = gmg_agg_l0_size;
    config.gmg.symmetric = symmetric;
    config.gmg.bottomSolver = gmg_bottom_solver;
    config.gmg.bottomMaxIter = gmg_bottom_max_iter;
    config.gmg.bottomRtol = gmg_bottom_rtol;
    config.mpInnerRtol = mp_inner_rtol;
    config.mpInnerMaxIter = mp_inner_max_iter;
    config.norm = norm;
    config.bcData = bc_data;
    return config;
}

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
               std::optional<NeoN::Executor> executor,
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
               amrex::MultiFab* bc_data)
            {
                new (self)
                    S(executor.value_or(NeoN::createDefaultExecutor()),
                      geom,
                      &alpha,
                      &ux,
                      &lx,
                      &uy,
                      &ly,
                      &uz,
                      &lz,
                      parseSolverConfig(
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
                          gmg_coeff_precision,
                          gmg_omega,
                          gmg_agg_l0_size,
                          symmetric,
                          gmg_bottom_solver,
                          gmg_bottom_max_iter,
                          gmg_bottom_rtol,
                          mp_inner_rtol,
                          mp_inner_max_iter,
                          norm,
                          bc_data
                      ));
            },
            nb::arg("alpha"),
            nb::arg("ux"),
            nb::arg("lx"),
            nb::arg("uy"),
            nb::arg("ly"),
            nb::arg("uz"),
            nb::arg("lz"),
            nb::arg("geom"),
            nb::arg("executor") = nb::none(),
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
            //
            // solver="mpir" is mixed-precision iterative refinement, and needs
            // precond="gmg_kokkos" (the only preconditioner with an fp32 apply). The
            // OUTER loop is the same gko::solver::Ir<double> over the same fp64
            // FaceCoeffOp, so r = b - A x, the stopping test and the answer are the
            // fp64 solver's; what changes is the inner correction, a preconditioned
            // Cg<float>. Every Krylov vector in that inner solve is half the width,
            // which is the point: at 256^3 the Krylov side of a tuned solve is 74 of
            // 189 ms and every kernel in it is at 83-100% of the memory roofline, so
            // bytes are the only lever left.
            //
            // A MEASURED NEGATIVE RESULT, kept wired as one. The saving arrives: an
            // fp32 inner iteration costs 18.1 ms against fp64 CG's 21.4, i.e. the
            // 1.18x that halving the vectors predicts. Refinement then spends it and
            // more -- the cheapest schedule needs 15 preconditioner applies where CG
            // needed 10, because a restart re-pays the initial residual and the
            // pre-check preconditioner apply and discards the Krylov space. Net at
            // 256^3: 302 ms against 214, 1.41x SLOWER, and no inner tolerance
            // recovers it. Drive it with mp_inner_max_iter rather than
            // mp_inner_rtol. Full table and the two hypotheses ruled out in
            // mixedPrecision.hpp.
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
            // periodicity per direction. Both solvers: FaceCoeffSolver folds it
            // by ghost reflection, FaceCoeffCsrSolver by the equivalent
            // diagonal fold in the assembled entries.
            nb::arg("bc") = std::vector<std::string>(6, "periodic"),
            // Preconditioner selector: "none" (default; precond_mlmg alone
            // implies "mlmg"), "mlmg" (requires precond_mlmg) or "gmg" (native
            // matrix-free geometric multigrid on the face coefficients —
            // matrix-free solver only, no MLMG involved).
            nb::arg("precond") = "none",
            // Native-GMG (precond="gmg") V-cycle knobs. gmg_pre_sweeps/
            // gmg_post_sweeps: RB-GS sweep count / Chebyshev degree per pre-/
            // post-smooth (keep them equal for a CG-safe symmetric V-cycle).
            // gmg_coarsest_sweeps: smoothing on the bottom level. gmg_max_levels:
            // 0 = auto/unlimited coarsening; else cap the hierarchy depth.
            // gmg_min_bottom: stop coarsening before the domain shortside drops
            // below this. gmg_smoother: "rbgs" (red-black Gauss-Seidel) or
            // "chebyshev" (Jacobi-preconditioned polynomial, plain-stencil
            // bandwidth).
            //
            // gmg_min_bottom=2 and gmg_coarsest_sweeps=16 (with gmg_omega=1.1
            // below) are a MEASURED shape, not the historical one — the previous
            // 4/8/1.0 predated agglomeration, fp32, shared coefficients and
            // level-0 re-decomposition, four features that each changed what a
            // level costs. The three compose super-additively, because a deeper
            // ladder is what makes extra bottom sweeps cheap (a 2^3 bottom is 8
            // cells) and a better-solved bottom is what makes the deeper ladder
            // pay. Preconditioned CG on the periodic Helmholtz, fp32 hierarchy,
            // level-0 agglomerated:
            //
            //     grid     4/8/1.0        2/16/1.1      speedup
            //     64^3     11 iters        8 iters       1.23x
            //     128^3    11 iters        8 iters       1.28x
            //     256^3    12 iters        8 iters       1.40x
            //
            // The iteration count is now FLAT in N, which is the mesh-independence
            // multigrid is supposed to have and 4/8/1.0 did not. Confirmed off the
            // constant-coefficient problem at 128^3 (smooth b 12->8, a 1e4 b jump
            // 28->23, 4:1 anisotropic cells 81->59), and every knob is load-bearing
            // in the drop-one controls on every one of those.
            nb::arg("gmg_pre_sweeps") = 2,
            nb::arg("gmg_post_sweeps") = 2,
            nb::arg("gmg_coarsest_sweeps") = 16,
            nb::arg("gmg_max_levels") = 0,
            nb::arg("gmg_min_bottom") = 2,
            nb::arg("gmg_smoother") = "rbgs",
            // GMG hierarchy precision: "fp64" (default; byte-for-byte the
            // previous behaviour), "fp32" or "bf16" — the whole V-cycle (level
            // coefficients, work fields, smoother, restriction/prolongation,
            // ghost fills) is STORED in that type while the outer CG/operator
            // stays double, which is what shrinks the bandwidth-bound V-cycle
            // traffic: half at fp32, a quarter at bf16.
            //
            // "bf16" needs precond="gmg_kokkos" (the shipped GmgPrecondT
            // hierarchy is fp64/fp32); its arithmetic still happens in fp32,
            // only the stored values are 16-bit. It buys 1.36x on the V-cycle
            // and loses it, at every size measured, to the iteration count:
            // the restricted residual carries psi's ~0.4% storage error times
            // ||A|| ~ 6/dx^2, so the cycle weakens as n^2. 11 -> 25 CG
            // iterations at 64^3, 12 -> 273 at 256^3. A measured negative
            // result, wired up and kept as one: see bf16.hpp. Matrix-free
            // solver only.
            nb::arg("gmg_precision") = "fp64",
            // The storage type of the COEFFICIENTS alone (alpha and the face
            // arrays); "" (the default) means the same as gmg_precision, which is
            // what every level did before this option existed. May not be wider
            // than gmg_precision, and needs precond="gmg_kokkos".
            //
            // This is the half of the bf16 experiment that survives. A rounded psi
            // is amplified: the cycle restricts b - A psi, so psi's ~0.4% storage
            // error reaches the coarse grid times ||A|| ~ 6/dx^2, and the cycle
            // weakens as n^2 (see gmg_precision above). A rounded COEFFICIENT is a
            // ~0.4% perturbation of the preconditioner's operator and nothing
            // else -- the operator CG applies and the residual it stops on are
            // still fp64 -- so it cannot make an answer wrong, only cost
            // iterations. With gmg_share_coeffs on they are 4 of the 6 arrays a
            // colour sweep streams.
            //
            // Measured at 256^3 (single box, l0 agglomerated), fields/coeffs:
            //
            //   fp32/bf16   V-cycle 12.52 -> 10.60 ms, r1/r0 0.70185 -> 0.70147,
            //               CG 9 iterations either way, solve 213 -> 195 ms.
            //   fp64/bf16   23.82 -> 26.54 ms, i.e. 1.11x SLOWER at the same cycle
            //               strength -- narrowing pays only once the FIELDS are
            //               narrow. Kept reachable as the measured negative.
            //
            // Note the benchmark trap: on a constant-coefficient Laplacian over a
            // power-of-two grid every coefficient (1/dx^2, alpha=1, and the 1/4 and
            // 1/8 restriction weights) is exactly representable, so a bf16
            // hierarchy is BIT-IDENTICAL and reports the full saving for free. The
            // rows above use a varying b for exactly that reason.
            nb::arg("gmg_coeff_precision") = "",
            // RB-SOR relaxation factor for gmg_smoother="rbgs":
            //   sol <- sol + gmg_omega * (gs - sol)
            // 1.0 is plain red-black Gauss-Seidel; 1.1 (the default) over-relaxes
            // by 10%. Must lie in (0, 2) for a convergent relaxation. Ignored by
            // gmg_smoother="chebyshev", whose damping comes from the polynomial.
            //
            // Why over-relaxing helps a SMOOTHER, which is not trying to solve
            // anything: its whole job is to annihilate the modes the coarse grid
            // cannot represent, theta in [pi/2, pi]. At omega=1 RB-GS damps that
            // band very unevenly — modes near pi are crushed while modes near
            // pi/2, exactly where the coarse grid is also weakest, are barely
            // touched — and the cycle's contraction is set by the WORST mode in
            // the band. Over-relaxing trades surplus damping near pi for scarce
            // damping near pi/2, which lowers that maximum. MLMG's own abec_gsrb
            // over-relaxes for the same reason, at 1.15.
            //
            // It is bounded on the other side by symmetry: omega != 1.0 makes the
            // colour sweep non-self-adjoint, so the V-cycle is no longer exactly
            // SPD even with the reversed post-smooth, and CG's theory stops
            // applying. Harmless for solver="gmg"/"ir" (stationary iterations);
            // for precond="gmg" it is a real cost, and the measured turnover is
            // where it starts to outweigh the better damping. 256^3, preconditioned
            // CG, everything else at the defaults above:
            //
            //     omega    0.9   1.0   1.1   1.15   1.2   1.3
            //     iters     15    12    11     11    12    13
            //     (combined with min_bottom=2, coarsest_sweeps=16)
            //     iters      -    10     8      9     9     -
            //
            // Hence 1.1 rather than MLMG's 1.15, which costs an iteration at both
            // 128^3 and 256^3. The gain is largest where the coarse grid is worst:
            // 4:1 anisotropic cells (we coarsen all three axes, no semicoarsening)
            // go 70 -> 59 iterations, against 12 -> 11 on the isotropic problem.
            //
            // WARNING: this is the V-cycle's SECOND symmetry breaker, and the two
            // do not compose. gmg_pre_sweeps != gmg_post_sweeps is the first, and
            // it already warns. Either alone is survivable; both at once are not
            // (N=32, precond="gmg", 300-iteration budget):
            //
            //     sweeps    omega=1.0   omega=1.05   omega=1.1   omega=1.15
            //     2 / 1      16 iters    21 iters     diverges    diverges
            //     2 / 2       8 iters     8 iters      8 iters     8 iters
            //
            // Raising this default is safe precisely because the default sweeps
            // are symmetric. Set gmg_omega=1.0 whenever they are not. Pinned by
            // test_asymmetric_sweeps_and_over_relaxation_stack.
            nb::arg("gmg_omega") = 1.1,
            // Target box size for LEVEL 0 of the gmg_kokkos hierarchy; 0 (the default)
            // leaves level 0 on the caller's boxes, byte-for-byte the previous
            // behaviour. Level 0 holds 7/8 of the hierarchy's cells and a box's halo
            // traffic falls as its side grows, so bigger boxes there are the single
            // largest remaining lever -- paid for with one copy per preconditioner
            // apply, since the solver's flat vectors are in the CALLER's cell order.
            // precond="gmg_kokkos" only; ignored by every other preconditioner.
            nb::arg("gmg_agg_l0_size") = 0,
            // Whether the caller declares the operator SYMMETRIC. Set explicitly
            // rather than sniffed from the coefficients: a set that happens to be
            // symmetric on this call may not be on the next, and silently switching
            // algorithm on that would change the answer without changing the
            // configuration. Symmetric is the default because the pressure Poisson
            // system this solver exists for is symmetric; convection makes it false.
            //
            // What it gates, all REFUSED rather than warned about, because none of
            // them fail loudly -- they converge to something wrong, or stall, and the
            // caller sees only a worse iteration count:
            //   * gmg_omega != 1.0        (over-relaxation's justification is a
            //                              self-adjointness argument)
            //   * gmg_smoother="chebyshev" (its polynomial is built on a REAL
            //                              eigenvalue interval; an asymmetric
            //                              operator has a complex spectrum)
            //   * gmg_bottom_solver="cg"/"fcg" (both need an SPD operator)
            // The outer solver is the caller's own choice and is NOT gated here:
            // solver="bicgstab" (the default) and "gmres" are already safe, and
            // ginkgo_solve_composite documents the same caveat for "cg".
            nb::arg("symmetric") = true,
            // How the COARSEST level is solved. "smoother" (the default) is the
            // historical behaviour: gmg_coarsest_sweeps smoother sweeps, no residual
            // test. It is cheap and, being fixed work, exactly stationary -- which
            // matters because the V-cycle is used as a CG preconditioner and CG
            // assumes a preconditioner that does not change between applies.
            //
            // It is also, on its own, unable to converge the bottom: a consistent
            // polynomial smoother has p(0) = 1, so the coarse grid's constant mode
            // survives every sweep no matter how many are run. MLMG solves its bottom
            // with a Krylov method for exactly this reason.
            //
            // "cg" | "fcg" | "bicgstab" | "gmres" | "gcr" generate the corresponding
            // Ginkgo solver on the coarsest level instead. cg/fcg need symmetric=True;
            // bicgstab/gmres/gcr do not. Prefer a TIGHT gmg_bottom_rtol: an adaptive
            // bottom makes the V-cycle a different operator on each apply, which an
            // outer Cg is not entitled to assume. Solve it nearly exactly (cheap --
            // the bottom is a handful of cells) or drive the outer solve with a
            // flexible method (solver="gcr" or "fcg").
            nb::arg("gmg_bottom_solver") = "smoother",
            nb::arg("gmg_bottom_max_iter") = 200,
            nb::arg("gmg_bottom_rtol") = 1e-12,
            // solver="mpir" only. The relative residual the INNER fp32 Cg stops at,
            // and its iteration cap. This is the whole design of a refinement
            // scheme: the outer contraction factor IS the inner tolerance, so 1e-2
            // means ~2 digits per outer step. Tighter wastes fp32 iterations on
            // digits the outer loop recomputes; looser turns the outer loop into
            // Richardson. Ignored by every other solver.
            nb::arg("mp_inner_rtol") = 1e-2,
            nb::arg("mp_inner_max_iter") = 20,
            // Which norm the stopping test (and the reported res_norm) measures:
            // "l2" (default, Ginkgo's ||r||_2 <= rtol*||b||_2 — byte-for-byte the
            // previous behaviour) or "linf", AMReX MLMG's criterion
            // ||r||_inf <= rtol*||b||_inf. Two solvers stopping on different norms
            // are answering different questions, so "linf" is what makes an
            // iteration count directly comparable with mlmg's. Applies to the
            // Krylov path (via a custom Ginkgo criterion) and to the native
            // stationary solver="gmg" loop alike.
            nb::arg("norm") = "l2",
            // INHOMOGENEOUS domain BC data, or None (the default) for the
            // homogeneous fills `bc` alone gives. A cell-centred MultiFab on
            // alpha's BoxArray/DistributionMapping with >= 1 ghost cell, carrying
            // the boundary datum in its GHOST layer — MLMG's set_level_bc
            // contract, so one MultiFab drives both solvers and the two can be
            // compared directly. Per boundary face:
            //   'dirichlet' side -> u ON the face
            //   'neumann'   side -> du/dn, the OUTWARD normal derivative
            // Only ghost cells outside a non-periodic domain face are read; the
            // valid region and the periodic/internal ghosts are ignored, so the
            // same fab may be the solution's own ghosted MultiFab.
            //
            // The data is REFERENCED, not copied (device path), so an in-place
            // update takes effect on the next solve — the coefficient contract.
            //
            // What it costs: with inhomogeneous BCs the boundary operator is
            // AFFINE, L(x) = A x + c0, and Ginkgo's Krylov solvers assume a
            // linear one. So the Krylov path solves A x = rhs - c0 with `apply`
            // still computing A alone, paying ONE extra apply per solve to form
            // c0 = L(0), plus one n-sized vector to hold it. solver='gmg' pays
            // neither: its outer residual can be rhs - L(x) directly, and the
            // V-cycle underneath still solves for a correction, whose boundary
            // condition is homogeneous whatever the solution's is.
            //
            // Matrix-free solver only; needs at least one non-periodic side.
            nb::arg("bc_data").none() = nb::none(),
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
            nb::keep_alive<1, 16>(),
            // bc_data (arg 37, the last one) is referenced by the operator on the
            // device path exactly as the coefficients are.
            nb::keep_alive<1, 37>()
        )
        .def(
            "solve",
            [](S& self, amrex::MultiFab& rhs, amrex::MultiFab& sol)
            { return toDict(self.solve(rhs, sol)); },
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
            "kwarg, both solvers): 6 entries (xlo, xhi, ylo, yhi,\n"
            "zlo, zhi) of 'periodic' | 'dirichlet' | 'neumann' — homogeneous\n"
            "domain BCs folded in via ghost reflection; must match the\n"
            "geometry's periodicity per direction. bc_data (constructor kwarg,\n"
            "matrix-free solver only) makes those BCs INHOMOGENEOUS: a ghosted\n"
            "cell MultiFab whose ghost layer holds u on the face (dirichlet\n"
            "sides) or du/dn outward (neumann sides), read fresh each solve.\n"
            "Returns a\n"
            "dict with num_iters, res_norm, converged and res_history (per-\n"
            "iteration residual norms of this call)."
        );
}

// --- The blockamr::la matrix formats (S4) ----------------------------------
//
// blockAmr has NO C++ test target: NeoN_BUILD_TESTS is OFF for the Python build
// and test/CMakeLists.txt adds no blockAmr subdirectory, so a C++ test would
// never be compiled. Everything below therefore exists to make la::Matrix,
// la::MFFaceCoeffs and la::CsrMatrix reachable from pytest, which is the only
// place this component is testable at all.
//
// It is deliberately NOT the S5 Solver facade: no SolverConfig surface, no
// LinearSystem, no operators. It builds a format through its factory, writes the
// caller's coefficients through Matrix::coefficients(), and drives Matrix::op()
// with the existing KrylovSolver so the two formats can be compared against a
// hand-built FaceCoeffSolver on the same problem. When S5 lands its own solver,
// this goes.

// The one thing missing to run a Matrix through the existing Krylov machinery:
// KrylovSolver builds itself from a LinOp, and Matrix::op() is one.
class MatrixKrylovSolver : public KrylovSolver
{
public:

    MatrixKrylovSolver(
        const blockamr::la::Matrix& matrix,
        gko::size_type nGlobal,
        const std::string& solver,
        int maxIter,
        double rtol,
        double atol,
        bool projectNullspace
    )
        : KrylovSolver(makeExecutor(matrix.executor()), nGlobal, matrix.localRows())
    {
        // const_pointer_cast because build() takes a mutable LinOp (it hands it to
        // Ginkgo's solver factories, which store a non-const system matrix); op()
        // is const-correct on the format's side and nothing here writes through it.
        build(
            std::const_pointer_cast<gko::LinOp>(matrix.op()),
            solver,
            maxIter,
            rtol,
            atol,
            projectNullspace
        );
    }
};

blockamr::la::Matrix makeLaMatrix(
    const std::string& format,
    const std::string& symmetry,
    const NeoN::Executor& nexec,
    const amrex::MultiFab& like,
    const amrex::Geometry& geom,
    const BcArray& bc
)
{
    const amrex::BoxArray& ba = like.boxArray();
    const amrex::DistributionMapping& dm = like.DistributionMap();
    const bool sym = (symmetry == "symmetric");
    if (symmetry != "symmetric" && symmetry != "asymmetric")
    {
        throw std::runtime_error(
            "la matrix: unknown symmetry '" + symmetry + "' (expected 'symmetric' or 'asymmetric')"
        );
    }
    if (format == "mf")
    {
        return sym ? blockamr::la::Matrix(
                   blockamr::la::MFFaceCoeffs::symmetric(nexec, ba, dm, geom, bc)
               )
                   : blockamr::la::Matrix(
                       blockamr::la::MFFaceCoeffs::asymmetric(nexec, ba, dm, geom, bc)
                   );
    }
    if (format == "csr")
    {
        return sym
                 ? blockamr::la::Matrix(blockamr::la::CsrMatrix::symmetric(nexec, ba, dm, geom, bc))
                 : blockamr::la::Matrix(blockamr::la::CsrMatrix::asymmetric(nexec, ba, dm, geom, bc)
                 );
    }
    throw std::runtime_error("la matrix: unknown format '" + format + "' (expected 'mf' or 'csr')");
}

// Copy one caller-supplied field into the field the format allocated for it. The
// layout check is explicit rather than left to AMREX_ASSERT, which is compiled
// out in a Release build -- a silent mismatch here would read as a solver bug.
void writeField(amrex::MultiFab& dst, const amrex::MultiFab& src, const char* what)
{
    if (dst.boxArray() != src.boxArray() || dst.DistributionMap() != src.DistributionMap())
    {
        throw std::runtime_error(
            std::string("la matrix: ") + what + " has a different BoxArray/DistributionMapping "
            + "than the matrix allocated for it"
        );
    }
    amrex::MultiFab::Copy(dst, src, 0, 0, 1, 0);
}

// Fill a matrix from the seven caller fields, through Matrix::coefficients() and
// nothing else. l* are read only when the matrix is asymmetric -- a symmetric one
// reports an empty `lower` view, which IS the interface saying "there is no low
// side to write".
void writeCoefficients(
    blockamr::la::Matrix& matrix,
    const amrex::MultiFab& alpha,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz
)
{
    auto c = matrix.coefficients();
    // NOTE the negSumDiag convention: c.diag is the cell-centred diagonal SOURCE
    // alpha, not the matrix diagonal (which is alpha - sum(faces); since S7 the
    // matrix-free format stores that separately). See faceCoeffMatrix.hpp.
    writeField(*c.diag.ptr, alpha, "alpha");
    writeField(*c.upper.dir[0], ux, "ux");
    writeField(*c.upper.dir[1], uy, "uy");
    writeField(*c.upper.dir[2], uz, "uz");
    if (!c.lower.empty())
    {
        writeField(*c.lower.dir[0], lx, "lx");
        writeField(*c.lower.dir[1], ly, "ly");
        writeField(*c.lower.dir[2], lz, "lz");
    }
}

// --- S5: the LinearSystem / Operator / Solver seam ------------------------
//
// Same test-facing shape as the two entry points above, and for the same reason:
// blockAmr has no C++ test target that builds, so pytest is the only place this
// code is reachable at all. Underscore-prefixed, so `from ._blockamr import *`
// does not re-export them.

// The diagonal SOURCE alpha (ddt/Sp/reaction) has no operator yet -- ops::Ddt is a
// later slice -- so it is written straight through the matrix's own coefficient
// handles, exactly as S4's writeCoefficients does. The FACES are what
// `system += ops::Laplacian(...)` writes, and comparing them is what the tests do.
void writeDiagSource(blockamr::la::Matrix& matrix, const amrex::MultiFab& alpha)
{
    auto c = matrix.coefficients();
    writeField(*c.diag.ptr, alpha, "alpha");
}

// Copy the coefficients back out into caller-owned MultiFabs, so a test can
// compare them BITWISE against a hand-built set. Reads through the same
// Matrix::coefficients() handles an operator writes through.
void readCoefficients(
    blockamr::la::Matrix& matrix,
    amrex::MultiFab& alphaOut,
    amrex::MultiFab& uxOut,
    amrex::MultiFab& uyOut,
    amrex::MultiFab& uzOut
)
{
    auto c = matrix.coefficients();
    writeField(alphaOut, *c.diag.ptr, "alpha_out");
    writeField(uxOut, *c.upper.dir[0], "ux_out");
    writeField(uyOut, *c.upper.dir[1], "uy_out");
    writeField(uzOut, *c.upper.dir[2], "uz_out");
}

/* @brief What Python's `la.Solver` is: a parsed SolverConfig and nothing else.
 *
 * `blockamr::la::Solver` holds an executor AND a config, but the design's Python
 * surface builds one from a config alone -- the executor is given to the MATRIX
 * (`MFFaceCoeffs.symmetric(..., executor=exec)`) and reaches the solve through
 * `LinearSystem::executor()`. Carrying it twice would let the two disagree, and
 * the matrix's is the one that decides where the coefficient fields live. So the
 * la::Solver is built inside solve(), from the system.
 *
 * The `solver` / `precond` STRINGS are parsed here, once, at construction: this
 * is the Python boundary parseSolverConfig already is for FaceCoeffSolver, and
 * everything downstream compares SolverKind/PrecondKind enums (solverConfig.hpp).
 */
struct PyLaSolver
{
    SolverConfig cfg;

    SolveResult solve(const blockamr::la::LinearSystem& system, amrex::MultiFab& sol) const
    {
        return blockamr::la::Solver(system.executor(), cfg).solve(system, sol);
    }
};

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
           std::optional<NeoN::Executor> executor)
        { return toDict(solveMlmgSystem(lp, sol, rhs, max_iter, rtol, atol, sign, executor)); },
        nb::arg("lp"),
        nb::arg("sol"),
        nb::arg("rhs"),
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        nb::arg("atol") = 0.0,
        nb::arg("sign") = -1.0,
        nb::arg("executor") = nb::none(),
        "Matrix-free Ginkgo CG solve of the MLLinOp system L(sol) = rhs.\n\n"
        "sol's incoming values are the initial guess, and boundary data set\n"
        "via set_level_bc is honored (residual-correction solve). `sign` must\n"
        "make sign*L SPD: -1.0 (default) for MLPoisson (L = +laplacian,\n"
        "negative-definite); +1.0 for MLABecLaplacian (alpha*a*phi -\n"
        "beta*div(b grad phi), positive-definite). CG stops when\n"
        "||r_k|| <= rtol*||rhs|| (or ||r_k|| <= atol when atol > 0), so a warm\n"
        "start converges immediately.\n"
        "`executor` is a NeoN executor -- SerialExecutor (the default),\n"
        "CPUExecutor or GPUExecutor -- and selects the Ginkgo executor via\n"
        "NeoN.la.ginkgo.getGkoExecutor, so blockAMR and the rest of NeoN run on\n"
        "one memoized executor and one stream. On GPUExecutor the entire solve\n"
        "runs on the device: the Krylov vector ops, the\n"
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
           std::optional<NeoN::Executor> executor,
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
            return toDict(solveComposite(lp, sol, rhs, max_iter, rtol, atol, sign, executor, solver)
            );
        },
        nb::arg("lp"),
        nb::arg("sol"),
        nb::arg("rhs"),
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        nb::arg("atol") = 0.0,
        nb::arg("sign") = -1.0,
        nb::arg("executor") = nb::none(),
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
        "atol > 0). `executor` is a NeoN SerialExecutor / CPUExecutor /\n"
        "GPUExecutor. Returns a dict with\n"
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
            return toDict(solveFaceCoeffs(
                alpha, ux, lx, uy, ly, uz, lz, sol, rhs, geom, solver, max_iter, rtol
            ));
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
        "alpha - negSumDiag(faces), computed once when the operator is built.\n"
        "`solver` is one of\n"
        "'cg' (SPD only), 'bicgstab' (default), or 'gmres'. sol's incoming\n"
        "values seed the initial guess. CG/BiCGStab/GMRES stop when\n"
        "||r_k|| <= rtol*||rhs||. Returns a dict with num_iters and res_norm."
    );

    // Persistent solvers: build the operator + Ginkgo solver once, solve many
    // times. FaceCoeffSolver is matrix-free (recomputes the mat-vec from the
    // face coefficients each apply); FaceCoeffCsrSolver assembles the same
    // matrix into a CSR (single-box; homogeneous bc folded onto the diagonal)
    // so the benefit of matrix-free over an explicit sparse matrix can be
    // measured.
    bindPersistent<FaceCoeffSolver>(m, "FaceCoeffSolver");
    bindPersistent<FaceCoeffCsrSolver>(m, "FaceCoeffCsrSolver");

    // The blockamr::la matrix formats. Test-facing (see the comment above
    // MatrixKrylovSolver): the only way to reach S4's C++ from pytest, which is
    // the only place blockAmr is testable.
    m.def(
        "_la_matrix_solve",
        [](const std::string& format,
           MultiFab& alpha,
           MultiFab& ux,
           MultiFab& lx,
           MultiFab& uy,
           MultiFab& ly,
           MultiFab& uz,
           MultiFab& lz,
           const Geometry& geom,
           MultiFab& rhs,
           MultiFab& sol,
           std::optional<NeoN::Executor> executor,
           const std::string& symmetry,
           const std::string& solver,
           int max_iter,
           double rtol,
           double atol,
           bool project_nullspace,
           const std::vector<std::string>& bc,
           bool via_copy,
           bool assemble_before_write)
        {
            const NeoN::Executor nexec = executor.value_or(NeoN::createDefaultExecutor());
            const BcArray bcArr = parseBc(bc, geom, "_la_matrix_solve");
            auto matrix = makeLaMatrix(format, symmetry, nexec, alpha, geom, bcArr);
            if (assemble_before_write)
            {
                // Force an op() over the still-zero coefficients. An assembled
                // format that did not notice the write below would then solve with
                // that zero matrix; a matrix-free one cannot go stale this way.
                auto stale = matrix.op();
                (void)stale;
            }
            writeCoefficients(matrix, alpha, ux, lx, uy, ly, uz, lz);
            // The copy exercises Matrix's clone(): the copy shares the coefficient
            // fields (they are MultiFabs, which cannot be copied) and, for the
            // assembled format, the assembly-freshness state with them.
            blockamr::la::Matrix work = via_copy ? blockamr::la::Matrix(matrix) : std::move(matrix);
            MatrixKrylovSolver s(
                work,
                static_cast<gko::size_type>(alpha.boxArray().numPts()),
                solver,
                max_iter,
                rtol,
                atol,
                project_nullspace
            );
            nb::dict d = toDict(s.solve(rhs, sol));
            d["is_assembled"] = work.isAssembled();
            d["local_rows"] = work.localRows();
            d["symmetric"] = work.symmetry() == blockamr::la::Symmetry::symmetric;
            return d;
        },
        nb::arg("format"),
        nb::arg("alpha"),
        nb::arg("ux"),
        nb::arg("lx"),
        nb::arg("uy"),
        nb::arg("ly"),
        nb::arg("uz"),
        nb::arg("lz"),
        nb::arg("geom"),
        nb::arg("rhs"),
        nb::arg("sol"),
        nb::arg("executor") = nb::none(),
        nb::arg("symmetry") = "symmetric",
        nb::arg("solver") = "cg",
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        nb::arg("atol") = 0.0,
        nb::arg("project_nullspace") = false,
        nb::arg("bc") = std::vector<std::string>(6, "periodic"),
        nb::arg("via_copy") = false,
        nb::arg("assemble_before_write") = false,
        "Solve A sol = rhs through a blockamr::la::Matrix holding format='mf'\n"
        "(MFFaceCoeffs, matrix-free) or 'csr' (CsrMatrix, assembled).\n\n"
        "The matrix allocates its OWN coefficient fields; the seven passed here\n"
        "are copied into them through Matrix::coefficients(), so the write face\n"
        "is the interface's, not the solver's. With symmetry='symmetric' the\n"
        "`lower` view is empty and l* are ignored. via_copy=True solves through a\n"
        "COPY of the Matrix (its clone() path). assemble_before_write=True calls\n"
        "op() once before the coefficients are written, which an assembled format\n"
        "must notice. Returns the usual solve dict plus is_assembled, local_rows\n"
        "and symmetric."
    );
    m.def(
        "_la_matrix_probe",
        [](const std::string& format,
           MultiFab& alpha,
           MultiFab& ux,
           MultiFab& lx,
           MultiFab& uy,
           MultiFab& ly,
           MultiFab& uz,
           MultiFab& lz,
           const Geometry& geom,
           std::optional<NeoN::Executor> executor,
           const std::string& symmetry,
           const std::vector<std::string>& bc)
        {
            const NeoN::Executor nexec = executor.value_or(NeoN::createDefaultExecutor());
            const BcArray bcArr = parseBc(bc, geom, "_la_matrix_probe");
            auto matrix = makeLaMatrix(format, symmetry, nexec, alpha, geom, bcArr);

            nb::dict d;
            d["is_assembled"] = matrix.isAssembled();
            d["symmetric"] = matrix.symmetry() == blockamr::la::Symmetry::symmetric;
            d["local_rows"] = matrix.localRows();
            {
                auto c = matrix.coefficients();
                d["lower_empty"] = c.lower.empty();
                d["upper_empty"] = c.upper.empty();
                d["diag_empty"] = c.diag.empty();
                d["reports_symmetric"] = c.symmetric();
            }
            writeCoefficients(matrix, alpha, ux, lx, uy, ly, uz, lz);

            // Every op() below is held alive to the end of the scope, so two
            // distinct assemblies can never land on the same address.
            auto op1 = matrix.op();
            auto op2 = matrix.op();
            d["op_rows"] = static_cast<std::size_t>(op1->get_size()[0]);
            d["op_stable_without_write"] = (op1.get() == op2.get());
            {
                auto c = matrix.coefficients();
                (void)c;
            }
            auto op3 = matrix.op();
            d["op_rebuilt_after_coefficients"] = (op3.get() != op2.get());
            matrix.zero();
            auto op4 = matrix.op();
            d["op_rebuilt_after_zero"] = (op4.get() != op3.get());
            return d;
        },
        nb::arg("format"),
        nb::arg("alpha"),
        nb::arg("ux"),
        nb::arg("lx"),
        nb::arg("uy"),
        nb::arg("ly"),
        nb::arg("uz"),
        nb::arg("lz"),
        nb::arg("geom"),
        nb::arg("executor") = nb::none(),
        nb::arg("symmetry") = "symmetric",
        nb::arg("bc") = std::vector<std::string>(6, "periodic"),
        "Introspect a blockamr::la::Matrix without solving: isAssembled,\n"
        "symmetry, localRows, which coefficient views are empty, the row count of\n"
        "the operator op() hands back, and whether op() is rebuilt after a write\n"
        "through coefficients() / zero() or reused when nothing was written."
    );

    // The S5 seam: LinearSystem + Operator + Solver + ops::Laplacian.
    m.def(
        "_la_system_solve",
        [](const std::string& format,
           MultiFab& gamma,
           MultiFab& alpha,
           const Geometry& geom,
           MultiFab& rhs,
           MultiFab& sol,
           MultiFab& alpha_out,
           MultiFab& ux_out,
           MultiFab& uy_out,
           MultiFab& uz_out,
           std::optional<NeoN::Executor> executor,
           const std::string& symmetry,
           const std::string& solver,
           int max_iter,
           double rtol,
           double atol,
           bool project_nullspace,
           const std::vector<std::string>& bc,
           const MultiFab* bc_data)
        {
            const NeoN::Executor nexec = executor.value_or(NeoN::createDefaultExecutor());
            const BcArray bcArr = parseBc(bc, geom, "_la_system_solve");
            auto matrix = makeLaMatrix(format, symmetry, nexec, gamma, geom, bcArr);
            writeDiagSource(matrix, alpha);

            blockamr::la::LinearSystem system(matrix, rhs);
            system += blockamr::la::Operator(blockamr::ops::Laplacian(gamma, geom, bcArr, bc_data));
            readCoefficients(matrix, alpha_out, ux_out, uy_out, uz_out);

            SolverConfig cfg;
            cfg.solver = solver;
            cfg.solverKind = parseSolverKind(solver);
            cfg.maxIter = max_iter;
            cfg.rtol = rtol;
            cfg.atol = atol;
            cfg.projectNullspace = project_nullspace;

            blockamr::la::Solver s(nexec, cfg);
            nb::dict d = toDict(s.solve(system, sol));
            d["is_assembled"] = system.matrix().isAssembled();
            d["local_rows"] = system.localRows();
            d["symmetric"] = system.matrix().symmetry() == blockamr::la::Symmetry::symmetric;
            // Non-owning: the rhs the solve reads IS the caller's MultiFab.
            d["rhs_aliases_input"] = (&system.rhs() == &rhs);
            return d;
        },
        nb::arg("format"),
        nb::arg("gamma"),
        nb::arg("alpha"),
        nb::arg("geom"),
        nb::arg("rhs"),
        nb::arg("sol"),
        nb::arg("alpha_out"),
        nb::arg("ux_out"),
        nb::arg("uy_out"),
        nb::arg("uz_out"),
        nb::arg("executor") = nb::none(),
        nb::arg("symmetry") = "symmetric",
        nb::arg("solver") = "cg",
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        nb::arg("atol") = 0.0,
        nb::arg("project_nullspace") = false,
        nb::arg("bc") = std::vector<std::string>(6, "periodic"),
        nb::arg("bc_data").none() = nb::none(),
        "Assemble A through `system += ops::Laplacian(gamma, geom, bc)` and solve\n"
        "A sol = rhs with blockamr::la::Solver.\n\n"
        "`alpha` is the cell-centred diagonal SOURCE and is written through\n"
        "Matrix::coefficients() directly -- there is no ops::Ddt yet. The FACE\n"
        "coefficients are the operator's alone, as is the BC fold: a non-periodic\n"
        "side gets a zero face coefficient and (sign-1)*aF on alpha, and with\n"
        "bc_data set also -aF*scale*g on the rhs -- which MUTATES the rhs passed\n"
        "in, since LinearSystem is non-owning. The assembled coefficients are\n"
        "copied back into alpha_out/u{x,y,z}_out so a caller can compare them\n"
        "bitwise with a hand-built set. Returns the usual solve dict plus\n"
        "is_assembled, local_rows, symmetric and rhs_aliases_input."
    );
    m.def(
        "_la_system_probe",
        [](const std::string& format,
           MultiFab& gamma,
           MultiFab& alpha,
           const Geometry& geom,
           MultiFab& rhs,
           MultiFab& alpha_out,
           MultiFab& ux_out,
           MultiFab& uy_out,
           MultiFab& uz_out,
           std::optional<NeoN::Executor> executor,
           const std::string& symmetry,
           const std::vector<std::string>& bc,
           int n_apply,
           bool zero_after,
           const MultiFab* bc_data,
           bool report_structure)
        {
            const NeoN::Executor nexec = executor.value_or(NeoN::createDefaultExecutor());
            const BcArray bcArr = parseBc(bc, geom, "_la_system_probe");
            auto matrix = makeLaMatrix(format, symmetry, nexec, gamma, geom, bcArr);
            writeDiagSource(matrix, alpha);

            blockamr::la::LinearSystem system(matrix, rhs);
            for (int i = 0; i < n_apply; ++i)
            {
                system +=
                    blockamr::la::Operator(blockamr::ops::Laplacian(gamma, geom, bcArr, bc_data));
            }
            if (zero_after)
            {
                system.zero();
            }
            readCoefficients(matrix, alpha_out, ux_out, uy_out, uz_out);

            nb::dict d;
            d["is_assembled"] = system.matrix().isAssembled();
            d["local_rows"] = system.localRows();
            d["symmetric"] = system.matrix().symmetry() == blockamr::la::Symmetry::symmetric;
            d["rhs_aliases_input"] = (&system.rhs() == &rhs);
            d["rhs_sum"] = system.rhs().sum(0);
            if (report_structure)
            {
                // The SPARSITY of the assembled matrix, so a test can pin that a
                // non-periodic row drops its boundary column instead of carrying
                // an explicit 0.0 at the modular-wraparound one (sparse/csr.cpp
                // side(), S6a). That property is invisible to every solve-level
                // and coefficient-level check in this suite, because the two
                // spellings of the row are numerically identical.
                auto csr = std::dynamic_pointer_cast<const gko::matrix::Csr<double, int>>(
                    system.matrix().op()
                );
                if (csr == nullptr)
                {
                    throw std::runtime_error(
                        "_la_system_probe: report_structure needs format='csr' -- only the "
                        "assembled format has a row structure to report"
                    );
                }
                // Through the master executor so the arrays are host-readable on
                // any executor, not only the reference one.
                auto host = gko::clone(csr->get_executor()->get_master(), csr);
                const int* rowPtrs = host->get_const_row_ptrs();
                const int* colIdxs = host->get_const_col_idxs();
                const auto nRows = static_cast<std::size_t>(host->get_size()[0]);
                d["csr_row_ptrs"] = std::vector<int>(rowPtrs, rowPtrs + nRows + 1);
                d["csr_col_idxs"] =
                    std::vector<int>(colIdxs, colIdxs + host->get_num_stored_elements());
            }
            return d;
        },
        nb::arg("format"),
        nb::arg("gamma"),
        nb::arg("alpha"),
        nb::arg("geom"),
        nb::arg("rhs"),
        nb::arg("alpha_out"),
        nb::arg("ux_out"),
        nb::arg("uy_out"),
        nb::arg("uz_out"),
        nb::arg("executor") = nb::none(),
        nb::arg("symmetry") = "symmetric",
        nb::arg("bc") = std::vector<std::string>(6, "periodic"),
        nb::arg("n_apply") = 1,
        nb::arg("zero_after") = false,
        nb::arg("bc_data").none() = nb::none(),
        nb::arg("report_structure") = false,
        "Assemble a blockamr::la::LinearSystem without solving it.\n\n"
        "Applies ops::Laplacian `n_apply` times (operators ACCUMULATE, so twice is\n"
        "twice the coefficients) and optionally calls LinearSystem::zero()\n"
        "afterwards, which clears the coefficients AND the rhs. Copies the result\n"
        "into alpha_out/u{x,y,z}_out and reports local_rows, symmetric,\n"
        "is_assembled, rhs_aliases_input and the rhs sum.\n\n"
        "report_structure=True additionally calls op() and returns the assembled\n"
        "matrix's csr_row_ptrs and csr_col_idxs (host copies), so a test can check\n"
        "the SPARSITY of a boundary row. format='csr' only."
    );

    // The S7 seam: MFFaceCoeffs' stored fine-level diagonal. Reached through the
    // concrete format rather than through Matrix, because the diagonal is
    // deliberately NOT part of MatrixCoefficients (whose `diag` is still alpha,
    // the source) and so the erasure cannot see it.
    m.def(
        "_la_stored_diagonal",
        [](MultiFab& alpha,
           MultiFab& ux,
           MultiFab& lx,
           MultiFab& uy,
           MultiFab& ly,
           MultiFab& uz,
           MultiFab& lz,
           const Geometry& geom,
           MultiFab& diag_out,
           MultiFab* alpha2,
           MultiFab* diag2_out,
           MultiFab* diag_zero_out,
           std::optional<NeoN::Executor> executor,
           const std::string& symmetry,
           const std::vector<std::string>& bc,
           bool rewrite_through_copy)
        {
            const NeoN::Executor nexec = executor.value_or(NeoN::createDefaultExecutor());
            const BcArray bcArr = parseBc(bc, geom, "_la_stored_diagonal");
            const amrex::BoxArray& ba = alpha.boxArray();
            const amrex::DistributionMapping& dm = alpha.DistributionMap();
            if (symmetry != "symmetric" && symmetry != "asymmetric")
            {
                throw std::runtime_error("la matrix: unknown symmetry '" + symmetry + "'");
            }
            auto m = (symmetry == "symmetric")
                       ? blockamr::la::MFFaceCoeffs::symmetric(nexec, ba, dm, geom, bcArr)
                       : blockamr::la::MFFaceCoeffs::asymmetric(nexec, ba, dm, geom, bcArr);
            {
                auto c = m.coefficients();
                writeField(*c.diag.ptr, alpha, "alpha");
                writeField(*c.upper.dir[0], ux, "ux");
                writeField(*c.upper.dir[1], uy, "uy");
                writeField(*c.upper.dir[2], uz, "uz");
                if (!c.lower.empty())
                {
                    writeField(*c.lower.dir[0], lx, "lx");
                    writeField(*c.lower.dir[1], ly, "ly");
                    writeField(*c.lower.dir[2], lz, "lz");
                }
            }
            writeField(diag_out, m.diagonal(), "diag_out");

            if (alpha2 != nullptr)
            {
                if (diag2_out == nullptr)
                {
                    throw std::runtime_error("_la_stored_diagonal: alpha2 needs diag2_out");
                }
                // A copy shares the fields AND the diagonal state, so a write
                // through it must be visible to the diagonal the ORIGINAL hands
                // out. That is the whole reason the state is shared.
                if (rewrite_through_copy)
                {
                    blockamr::la::MFFaceCoeffs copy = m;
                    auto c = copy.coefficients();
                    writeField(*c.diag.ptr, *alpha2, "alpha2");
                }
                else
                {
                    auto c = m.coefficients();
                    writeField(*c.diag.ptr, *alpha2, "alpha2");
                }
                writeField(*diag2_out, m.diagonal(), "diag2_out");
            }
            if (diag_zero_out != nullptr)
            {
                m.zero();
                writeField(*diag_zero_out, m.diagonal(), "diag_zero_out");
            }
        },
        nb::arg("alpha"),
        nb::arg("ux"),
        nb::arg("lx"),
        nb::arg("uy"),
        nb::arg("ly"),
        nb::arg("uz"),
        nb::arg("lz"),
        nb::arg("geom"),
        nb::arg("diag_out"),
        nb::arg("alpha2").none() = nb::none(),
        nb::arg("diag2_out").none() = nb::none(),
        nb::arg("diag_zero_out").none() = nb::none(),
        nb::arg("executor") = nb::none(),
        nb::arg("symmetry") = "symmetric",
        nb::arg("bc") = std::vector<std::string>(6, "periodic"),
        nb::arg("rewrite_through_copy") = false,
        "Copy out MFFaceCoeffs' stored fine-level diagonal, alpha - sum(faces).\n\n"
        "Writes the seven coefficients through Matrix-style coefficients() handles\n"
        "and copies the resulting diagonal into diag_out. With alpha2/diag2_out,\n"
        "rewrites the diagonal SOURCE afterwards and copies the refreshed diagonal\n"
        "too (rewrite_through_copy routes that write through a COPY of the format,\n"
        "which shares the freshness state). With diag_zero_out, calls zero() last\n"
        "and copies the diagonal that follows."
    );

    // ------------------------------------------------------------------
    // S8: blockamr::la as REAL Python classes, driving the same machinery the
    // underscore-prefixed test seams above drive. Those stay -- the S4/S5/S6b
    // tests are written against them, and they reach shapes (probes, bitwise
    // coefficient copy-outs) this surface deliberately does not expose.
    //
    // `blockamr.linear_algebra` is the module that gives these their public
    // names; the format classes MFFaceCoeffs / CsrMatrix live there, because at
    // this level a format is a FACTORY and what it hands back is the erased
    // Matrix (linearAlgebra/matrix.hpp).
    // ------------------------------------------------------------------

    // Each factory is its own binding rather than one taking a format string:
    // the whole point of the erasure is that the format is chosen once, on one
    // line, and never dispatched on again.
    const auto matrixFactory = [](auto make)
    {
        return [make](
                   const BoxArray& ba,
                   const DistributionMapping& dm,
                   const Geometry& geom,
                   std::optional<NeoN::Executor> executor,
                   const std::vector<std::string>& bc
               )
        {
            const NeoN::Executor nexec = executor.value_or(NeoN::createDefaultExecutor());
            return blockamr::la::Matrix(make(nexec, ba, dm, geom, parseBc(bc, geom, "la matrix")));
        };
    };
    nb::class_<blockamr::la::Matrix>(m, "Matrix")
        .def_static(
            "mf_symmetric",
            matrixFactory([](auto&&... a) { return blockamr::la::MFFaceCoeffs::symmetric(a...); }),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("geom"),
            nb::arg("executor") = nb::none(),
            nb::arg("bc") = std::vector<std::string>(6, "periodic")
        )
        .def_static(
            "mf_asymmetric",
            matrixFactory([](auto&&... a) { return blockamr::la::MFFaceCoeffs::asymmetric(a...); }),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("geom"),
            nb::arg("executor") = nb::none(),
            nb::arg("bc") = std::vector<std::string>(6, "periodic")
        )
        .def_static(
            "csr_symmetric",
            matrixFactory([](auto&&... a) { return blockamr::la::CsrMatrix::symmetric(a...); }),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("geom"),
            nb::arg("executor") = nb::none(),
            nb::arg("bc") = std::vector<std::string>(6, "periodic")
        )
        .def_static(
            "csr_asymmetric",
            matrixFactory([](auto&&... a) { return blockamr::la::CsrMatrix::asymmetric(a...); }),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("geom"),
            nb::arg("executor") = nb::none(),
            nb::arg("bc") = std::vector<std::string>(6, "periodic")
        )
        .def("is_assembled", &blockamr::la::Matrix::isAssembled)
        .def("local_rows", &blockamr::la::Matrix::localRows)
        .def(
            "is_symmetric",
            [](const blockamr::la::Matrix& mat)
            { return mat.symmetry() == blockamr::la::Symmetry::symmetric; }
        )
        .def("zero", &blockamr::la::Matrix::zero)
        .def(
            "diagonal_source",
            [](blockamr::la::Matrix& mat, const MultiFab& alpha) { writeDiagSource(mat, alpha); },
            nb::arg("alpha"),
            "Write the cell-centred diagonal SOURCE (ddt/Sp/reaction) straight\n"
            "through Matrix::coefficients(). There is no ops::Ddt yet, so this is\n"
            "the only way to set it; it is NOT the matrix diagonal, which stays\n"
            "alpha - sum(faces)."
        );

    // Opaque on purpose: an Operator's whole surface is `assemble`, which is
    // private with LinearSystem as its only friend (operator.hpp). `system += op`
    // is the only thing a caller can do with one, here as in C++.
    nb::class_<blockamr::la::Operator>(m, "Operator");

    m.def(
        "la_laplacian",
        [](const MultiFab& gamma,
           const Geometry& geom,
           const std::vector<std::string>& bc,
           const MultiFab* bcData)
        {
            const BcArray bcArr = parseBc(bc, geom, "laplacian");
            return blockamr::la::Operator(blockamr::ops::Laplacian(gamma, geom, bcArr, bcData));
        },
        nb::arg("gamma"),
        nb::arg("geom"),
        nb::arg("bc") = std::vector<std::string>(6, "periodic"),
        nb::arg("bc_data").none() = nb::none(),
        // gamma and bc_data are held BY POINTER (laplacian.hpp, LIFETIME) and
        // read when `system +=` runs, which may be much later.
        nb::keep_alive<0, 1>(),
        nb::keep_alive<0, 4>(),
        "The implicit diffusion term as face coefficients: `system += laplacian(...)`.\n\n"
        "Named `la_laplacian` here because `blockamr.laplacian` is already the\n"
        "stencil-kernel binding; `blockamr.linear_algebra.laplacian` is the name\n"
        "callers use."
    );

    nb::class_<blockamr::la::LinearSystem>(m, "LinearSystem")
        .def(
            nb::init<blockamr::la::Matrix&, MultiFab&>(),
            nb::arg("matrix"),
            nb::arg("rhs"),
            // Non-owning by design: both must outlive the system, and the rhs an
            // operator writes IS the caller's field.
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>()
        )
        .def(
            "__iadd__",
            [](blockamr::la::LinearSystem& s,
               const blockamr::la::Operator& op) -> blockamr::la::LinearSystem&
            {
                s += op;
                return s;
            },
            nb::arg("op"),
            nb::rv_policy::none
        )
        .def("zero", &blockamr::la::LinearSystem::zero)
        .def("local_rows", &blockamr::la::LinearSystem::localRows)
        .def("matrix", &blockamr::la::LinearSystem::matrix, nb::rv_policy::reference_internal)
        .def("rhs", &blockamr::la::LinearSystem::rhs, nb::rv_policy::reference_internal);

    nb::class_<PyLaSolver>(m, "Solver")
        .def(
            "__init__",
            [](PyLaSolver* self,
               const std::string& solver,
               int max_iter,
               double rtol,
               double atol,
               bool project_nullspace,
               const std::string& precond,
               const std::string& norm)
            {
                SolverConfig cfg;
                cfg.solver = solver;
                // The single parse, here at the Python boundary; every dispatch
                // downstream compares the enum (solverConfig.hpp).
                cfg.solverKind = parseSolverKind(solver);
                cfg.maxIter = max_iter;
                cfg.rtol = rtol;
                cfg.atol = atol;
                cfg.projectNullspace = project_nullspace;
                cfg.precond = precond;
                cfg.precondKind = parsePrecondKind(precond);
                cfg.norm = norm;
                new (self) PyLaSolver {cfg};
            },
            nb::arg("solver") = "bicgstab",
            nb::arg("max_iter") = 1000,
            nb::arg("rtol") = 1e-10,
            nb::arg("atol") = 0.0,
            nb::arg("project_nullspace") = false,
            nb::arg("precond") = "none",
            nb::arg("norm") = "l2"
        )
        .def(
            "solve",
            [](const PyLaSolver& s, const blockamr::la::LinearSystem& system, MultiFab& sol)
            { return toDict(s.solve(system, sol)); },
            nb::arg("system"),
            nb::arg("sol"),
            "Solve `system` into `sol` (in place; its incoming values seed the\n"
            "initial guess). The executor comes from the system's MATRIX.\n\n"
            "precond != 'none' and solver in ('gmg', 'ir', 'mpir') RAISE here --\n"
            "the GMG hierarchy is built from the coefficient fields rather than\n"
            "from a LinOp, so it is not reachable through this path. The error\n"
            "explains that; it is not a bug to work around."
        );

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
