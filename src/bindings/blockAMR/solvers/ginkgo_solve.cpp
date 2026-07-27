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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../bindings.hpp"

#include "../../../blockAmrSolvers/common/profiling.hpp"
#include "../../../blockAmrSolvers/common/transfer.hpp"
#include "../../../blockAmrSolvers/common/types.hpp"
#include "../../../blockAmrSolvers/krylov/executor.hpp"
#include "../../../blockAmrSolvers/krylov/krylov.hpp"
#include "../../../blockAmrSolvers/krylov/logging.hpp"
#include "../../../blockAmrSolvers/operators/face_coeff_op.hpp"
#include "../../../blockAmrSolvers/operators/mlmg_ops.hpp"
#include "persistent.hpp"

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
                      bc_data);
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
            // mixed_precision.hpp.
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
            // Which norm the stopping test (and the reported res_norm) measures:
            // "l2" (default, Ginkgo's ||r||_2 <= rtol*||b||_2 — byte-for-byte the
            // previous behaviour) or "linf", AMReX MLMG's criterion
            // ||r||_inf <= rtol*||b||_inf. Two solvers stopping on different norms
            // are answering different questions, so "linf" is what makes an
            // iteration count directly comparable with mlmg's. Applies to the
            // Krylov path (via a custom Ginkgo criterion) and to the native
            // stationary solver="gmg" loop alike.
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
            "geometry's periodicity per direction. bc_data (constructor kwarg,\n"
            "matrix-free solver only) makes those BCs INHOMOGENEOUS: a ghosted\n"
            "cell MultiFab whose ghost layer holds u on the face (dirichlet\n"
            "sides) or du/dn outward (neumann sides), read fresh each solve.\n"
            "Returns a\n"
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
           std::optional<NeoN::Executor> executor)
        {
            MLMG mlmg(lp);

            // A SerialExecutor keeps the Krylov vector ops on the CPU; a
            // GPUExecutor runs them on the device. The mat-vec (MLMG::apply) is
            // on the GPU either way. None means SerialExecutor here -- the
            // default is resolved at CALL time, not at binding-registration
            // time, so importing blockamr does not require neon to have been
            // imported first (converting a NeoN::Executor default needs _neon's
            // nb::class_ registrations, and getting that order wrong raises
            // std::bad_cast at import).
            auto exec = makeExecutor(executor.value_or(NeoN::Executor {NeoN::SerialExecutor {}}));
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

            MLMG mlmg(lp);

            auto exec = makeExecutor(executor.value_or(NeoN::Executor {NeoN::SerialExecutor {}}));

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
