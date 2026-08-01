// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Nanobind bindings for the blockAmr linear-algebra layer (include/NeoN/blockAmr/linearAlgebra):
// the matrix formats, LinearSystem, the solvers and the one-shot solve entry points. Ginkgo is
// what they are BUILT on, not what they expose, so the file is named for the layer.
//
// The MLLinOp mat-vec (MLMG::apply) is AFFINE, not linear, so those solves run in
// residual-correction form: A_home(delta) = sign*(rhs - L_inhom(x0)), sol = x0 + delta --
// `sign` per the docstrings.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
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
#include "NeoN/blockAmr/core/meshLevel.hpp"
#include "NeoN/blockAmr/core/profiling.hpp"
#include "NeoN/blockAmr/core/types.hpp"
#include "NeoN/blockAmr/linearAlgebra/faceCoeffMatrix.hpp"
#include "NeoN/blockAmr/linearAlgebra/ginkgo/adapt.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/krylovSolver.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/result.hpp"
#include "NeoN/blockAmr/linearAlgebra/linearSystem.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.hpp"
#include "NeoN/blockAmr/linearAlgebra/solve/oneshot.hpp"
#include "NeoN/blockAmr/linearAlgebra/solve/persistent.hpp"
#include "NeoN/blockAmr/linearAlgebra/solver.hpp"
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"
#include "NeoN/blockAmr/operators/laplacian.hpp"

namespace nb = nanobind;

using namespace blockamr::la;

namespace
{

// Every nb::arg default below reads from here, so solverConfig.hpp holds the one C++ default
// list and the two Python constructors cannot drift from it or from each other. Function-local
// so it is built on first use, not during this translation unit's static initialisation.
const SolverConfig& configDefaults()
{
    static const SolverConfig defaults {};
    return defaults;
}

// The one place a SolveResult crosses into Python. converged/res_history/contraction/
// diagnostic appear in the dict only when the corresponding field is set, which is what
// keeps ginkgo_solve_face_coeffs' 2-key surface (num_iters, res_norm) exactly as it was.
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

// Repackages the 27 non-fixed __init__ arguments into one SolverConfig, so the solvers take
// a `const SolverConfig&`. Also parses solver/precond to their enums, which throws on an
// unknown spelling -- before any constructor work runs (solverConfig.hpp).
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

// Bind a persistent solver class S. keep_alive ties the coefficient fields to the solver,
// since the matrix-free operator references them on the device.
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
            // Krylov "cg"|"bicgstab"|"gmres"|"gcr"|"fcg", or "gmg"/"ir": the native stationary
            // V-cycle loop and its Ginkgo Ir twin, which build the hierarchy, IGNORE `precond`
            // and want gmg_coarsest_sweeps raised. "mpir" needs precond="gmg_kokkos".
            nb::arg("solver") = configDefaults().solver,
            nb::arg("max_iter") = configDefaults().maxIter,
            nb::arg("rtol") = configDefaults().rtol,
            nb::arg("atol") = configDefaults().atol,
            nb::arg("project_nullspace") = configDefaults().projectNullspace,
            nb::arg("precond_mlmg").none() = nb::none(),
            nb::arg("precond_cycles") = configDefaults().precondCycles,
            // Domain BCs in order (xlo, xhi, ylo, yhi, zlo, zhi); each entry is "periodic",
            // "dirichlet" (homogeneous, u=0 on the face) or "neumann" (homogeneous, du/dn=0).
            // Must match the geometry's periodicity per direction. Both solvers.
            nb::arg("bc") = configDefaults().bc,
            // "none" (default; precond_mlmg alone implies "mlmg"), "mlmg" (requires
            // precond_mlmg) or "gmg" (native matrix-free GMG on the face coefficients, no
            // MLMG involved; matrix-free solver only).
            nb::arg("precond") = configDefaults().precond,
            // V-cycle knobs: pre/post_sweeps (RB-GS sweeps or Chebyshev degree per smooth,
            // keep equal for a CG-safe symmetric cycle), coarsest_sweeps (bottom smoothing),
            // max_levels (0 = unlimited), min_bottom (shortside floor), smoother rbgs|chebyshev.
            nb::arg("gmg_pre_sweeps") = configDefaults().gmg.preSweeps,
            nb::arg("gmg_post_sweeps") = configDefaults().gmg.postSweeps,
            // 2/16/1.1 is a MEASURED shape, flat in N where the historical 4/8/1.0 was not:
            // report/blockamr-precision-measurements.md#the-default-v-cycle-shape
            nb::arg("gmg_coarsest_sweeps") = configDefaults().gmg.coarsestSweeps,
            nb::arg("gmg_max_levels") = configDefaults().gmg.maxLevels,
            nb::arg("gmg_min_bottom") = configDefaults().gmg.minBottom,
            nb::arg("gmg_smoother") = configDefaults().gmg.smoother,
            // Storage type of the whole V-cycle: "fp32" (default), "fp64" (the historical
            // behaviour) or "bf16", which needs precond="gmg_kokkos" and is a measured negative
            // (bf16.hpp). Matrix-free solver only; the outer CG/operator/residual stay double.
            nb::arg("gmg_precision") = configDefaults().gmg.precision,
            // Storage type of the COEFFICIENTS alone; "" (default) means the same as
            // gmg_precision. May not be wider than gmg_precision, and needs
            // precond="gmg_kokkos". Measured: report/blockamr-precision-measurements.md
            nb::arg("gmg_coeff_precision") = configDefaults().gmg.coeffPrecision,
            // RB-SOR relaxation for gmg_smoother="rbgs": sol <- sol + gmg_omega*(gs - sol), in
            // (0, 2); ignored by "chebyshev". It breaks V-cycle symmetry, and DIVERGES combined
            // with gmg_pre_sweeps != gmg_post_sweeps -- use 1.0 whenever the sweeps differ.
            nb::arg("gmg_omega") = configDefaults().gmg.omega,
            // Target box size for LEVEL 0 of the gmg_kokkos hierarchy; 0 (default) leaves the
            // caller's boxes. Bigger boxes cut halo traffic but cost one copy per apply, since
            // the solver's flat vectors are in the CALLER's cell order. precond="gmg_kokkos".
            nb::arg("gmg_agg_l0_size") = configDefaults().gmg.aggLevel0Size,
            // Whether the caller DECLARES the operator symmetric (never sniffed). REFUSES, never
            // warns: gmg_omega != 1.0, gmg_smoother="chebyshev", gmg_bottom_solver="cg"/"fcg" --
            // all rest on self-adjointness. The outer solver is the caller's own and not gated.
            nb::arg("symmetric") = configDefaults().gmg.symmetric,
            // Coarsest-level solve. "smoother" (default) = gmg_coarsest_sweeps sweeps with no
            // residual test: cheap and exactly stationary (what a CG preconditioner needs) but
            // on its own unable to converge the bottom's constant mode, since p(0) = 1.
            nb::arg("gmg_bottom_solver") = configDefaults().gmg.bottomSolver,
            // "cg"/"fcg" (need symmetric=True) or "bicgstab"/"gmres"/"gcr" generate a Ginkgo
            // bottom solver instead, driven by these two knobs.
            nb::arg("gmg_bottom_max_iter") = configDefaults().gmg.bottomMaxIter,
            // Keep TIGHT: an adaptive bottom makes the V-cycle a different operator on each
            // apply, which an outer Cg may not assume -- or use solver="gcr"/"fcg".
            nb::arg("gmg_bottom_rtol") = configDefaults().gmg.bottomRtol,
            // solver="mpir" only: the relative residual the INNER fp32 Cg stops at, and its
            // iteration cap. The outer contraction factor IS the inner tolerance, so 1e-2 means
            // ~2 digits per outer step. Ignored by every other solver.
            nb::arg("mp_inner_rtol") = configDefaults().mpInnerRtol,
            nb::arg("mp_inner_max_iter") = configDefaults().mpInnerMaxIter,
            // Which norm the stopping test and the reported res_norm measure: "l2" (default,
            // ||r||_2 <= rtol*||b||_2) or "linf", MLMG's ||r||_inf <= rtol*||b||_inf -- what
            // makes an iteration count comparable with mlmg's. Krylov and solver="gmg" alike.
            nb::arg("norm") = configDefaults().norm,
            // INHOMOGENEOUS domain BC data, None (default) = homogeneous; layout and per-side
            // meaning in the `solve` docstring. REFERENCED, not copied (device path), so an
            // in-place update takes effect next solve. Matrix-free only; needs a non-periodic side.
            nb::arg("bc_data").none() = nb::none(),
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>(),
            nb::keep_alive<1, 4>(),
            nb::keep_alive<1, 5>(),
            nb::keep_alive<1, 6>(),
            nb::keep_alive<1, 7>(),
            nb::keep_alive<1, 8>(),
            // The preconditioner MLMG (arg 16; self=1, args from 2) must outlive the solver --
            // MlmgPrecond holds a raw pointer to it. A no-op when the arg is None.
            nb::keep_alive<1, 16>(),
            // bc_data (arg 37) is referenced by the operator on the device path, as the
            // coefficients are.
            nb::keep_alive<1, 37>()
        )
        .def(
            "solve",
            [](S& self, amrex::MultiFab& rhs, amrex::MultiFab& sol)
            { return toDict(self.solve(rhs, sol)); },
            nb::arg("rhs"),
            nb::arg("sol"),
            "Solve A sol = rhs, reusing the prebuilt operator and solver.\n\n"
            "sol's incoming values seed the initial guess. The matrix is defined by the\n"
            "coefficient fields handed to the constructor, which the matrix-free solver\n"
            "re-reads each call so in-place updates take effect. Constructor kwargs:\n"
            "project_nullspace=True (singular systems with the constant nullspace, e.g.\n"
            "fully-periodic pure Poisson) projects rhs and initial guess mean-zero and\n"
            "returns the mean-zero representative; precond_mlmg (an MLMG on an equivalent\n"
            "operator) preconditions each Krylov iteration with precond_cycles V-cycles;\n"
            "precond='gmg' (matrix-free solver only) uses the native V-cycle on the face\n"
            "coefficients instead, with no MLMG anywhere. bc is 6 entries (xlo, xhi, ylo,\n"
            "yhi, zlo, zhi) of 'periodic' | 'dirichlet' | 'neumann', homogeneous domain BCs\n"
            "folded in by ghost reflection; they must match the geometry's periodicity per\n"
            "direction. bc_data (matrix-free solver only) makes those BCs INHOMOGENEOUS: a\n"
            "MultiFab on alpha's BoxArray/DistributionMapping with >= 1 ghost cell, whose\n"
            "GHOST layer holds u ON the face (dirichlet sides) or du/dn OUTWARD (neumann\n"
            "sides), read fresh each solve; only ghosts outside a non-periodic domain face\n"
            "are read, so it may be the solution's own ghosted MultiFab.\n\n"
            "Returns a dict with num_iters, res_norm, converged and res_history (this\n"
            "call's per-iteration residual norms)."
        );
}

// blockAmr has NO C++ test target (NeoN_BUILD_TESTS is OFF for the Python build), so the
// bindings below exist to make la::MFFaceCoeffs reachable from pytest, which is the only place
// this component is testable at all.

// Runs a la::MFFaceCoeffs through the existing Krylov machinery: la::toLinOp gives it a LinOp.
class MatrixKrylovSolver : public KrylovSolver
{
public:

    MatrixKrylovSolver(
        const blockamr::la::MFFaceCoeffs& matrix,
        gko::size_type nGlobal,
        const std::string& solver,
        int maxIter,
        double rtol,
        double atol,
        bool projectNullspace
    )
        : KrylovSolver(makeExecutor(matrix.exec), nGlobal, matrix.localRows())
    {
        // const_pointer_cast: build() takes a mutable LinOp (Ginkgo's solver factories store
        // a non-const system matrix) and nothing here writes through it.
        build(
            std::const_pointer_cast<gko::LinOp>(toLinOp(matrix)),
            solver,
            maxIter,
            rtol,
            atol,
            projectNullspace
        );
    }
};

blockamr::la::MFFaceCoeffs makeLaMatrix(
    const std::string& symmetry,
    const NeoN::Executor& nexec,
    const amrex::MultiFab& like,
    const amrex::Geometry& geom,
    const BcArray& bc
)
{
    const blockamr::MeshLevel mesh {like.boxArray(), like.DistributionMap(), geom};
    if (symmetry != "symmetric" && symmetry != "asymmetric")
    {
        throw std::runtime_error(
            "la matrix: unknown symmetry '" + symmetry + "' (expected 'symmetric' or 'asymmetric')"
        );
    }
    return (symmetry == "symmetric") ? blockamr::la::MFFaceCoeffs::symmetric(nexec, mesh, bc)
                                     : blockamr::la::MFFaceCoeffs::asymmetric(nexec, mesh, bc);
}

// Copy one caller-supplied field into the field the format allocated for it. The layout check
// is explicit rather than AMREX_ASSERT, which is compiled out in a Release build.
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

// Fill a matrix from the seven caller fields through its coefficient fields and nothing else.
// l* are read only when the matrix is asymmetric; a symmetric one has no `lower` at all.
void writeCoefficients(
    blockamr::la::MFFaceCoeffs& matrix,
    const amrex::MultiFab& alpha,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz
)
{
    // negSumDiag convention: the matrix's `alpha` is the cell-centred diagonal SOURCE, not the
    // matrix diagonal alpha - sum(faces), which no format stores -- the stencil derives it.
    writeField(*matrix.alpha, alpha, "alpha");
    writeField(matrix.upper[0], ux, "ux");
    writeField(matrix.upper[1], uy, "uy");
    writeField(matrix.upper[2], uz, "uz");
    if (matrix.lower.has_value())
    {
        writeField((*matrix.lower)[0], lx, "lx");
        writeField((*matrix.lower)[1], ly, "ly");
        writeField((*matrix.lower)[2], lz, "lz");
    }
}

// The LinearSystem / Laplacian / Solver seam, test-facing for the same reason. Underscore-
// prefixed, so `from ._blockamr import *` does not re-export them.

// The diagonal SOURCE alpha (ddt/Sp/reaction) has no operator yet, so it is written straight
// into the matrix's own field; the FACES are what `system += ops::Laplacian` writes.
void writeDiagSource(blockamr::la::MFFaceCoeffs& matrix, const amrex::MultiFab& alpha)
{
    writeField(*matrix.alpha, alpha, "alpha");
}

// Copy the coefficients back out so a test can compare them BITWISE against a hand-built set.
// The three l*Out are OPTIONAL and written only when the matrix has a `lower` -- otherwise
// left exactly as passed, which lets a test assert nothing reached the low side.
void readCoefficients(
    const blockamr::la::MFFaceCoeffs& matrix,
    amrex::MultiFab& alphaOut,
    amrex::MultiFab& uxOut,
    amrex::MultiFab& uyOut,
    amrex::MultiFab& uzOut,
    amrex::MultiFab* lxOut = nullptr,
    amrex::MultiFab* lyOut = nullptr,
    amrex::MultiFab* lzOut = nullptr
)
{
    writeField(alphaOut, *matrix.alpha, "alpha_out");
    writeField(uxOut, matrix.upper[0], "ux_out");
    writeField(uyOut, matrix.upper[1], "uy_out");
    writeField(uzOut, matrix.upper[2], "uz_out");
    const bool wantLower =
        lxOut != nullptr && lyOut != nullptr && lzOut != nullptr && matrix.lower.has_value();
    if (wantLower)
    {
        writeField(*lxOut, (*matrix.lower)[0], "lx_out");
        writeField(*lyOut, (*matrix.lower)[1], "ly_out");
        writeField(*lzOut, (*matrix.lower)[2], "lz_out");
    }
}

/* @brief What Python's `la.Solver` is: a parsed SolverConfig and nothing else. The executor
 * belongs to the MATRIX and reaches the solve through `LinearSystem::executor()`, so carrying
 * it here too would let the two disagree; solver/precond are parsed here, once. */
struct PyLaSolver
{
    SolverConfig cfg;

    SolveResult solve(const blockamr::la::LinearSystem& system, amrex::MultiFab& sol) const
    {
        return blockamr::la::Solver(system.executor(), cfg).solve(system, sol);
    }
};

} // namespace

void registerLinearAlgebra(nb::module_& m)
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
        "sol's incoming values are the initial guess, and boundary data set via\n"
        "set_level_bc is honored (residual-correction solve). `sign` must make sign*L SPD:\n"
        "-1.0 (default) for MLPoisson (L = +laplacian, negative-definite); +1.0 for\n"
        "MLABecLaplacian (alpha*a*phi - beta*div(b grad phi), positive-definite). CG stops\n"
        "when ||r_k||_2 <= rtol*||rhs||_2 (or <= atol when atol > 0), so a warm start\n"
        "converges immediately. `executor` is a NeoN SerialExecutor (default), CPUExecutor\n"
        "or GPUExecutor, and selects the Ginkgo executor via NeoN.la.ginkgo.getGkoExecutor\n"
        "so blockAMR and the rest of NeoN share one memoized executor and one stream. On\n"
        "GPUExecutor the whole solve stays on the device, with no per-iteration host\n"
        "transfer.\n\n"
        "Returns a dict with num_iters, res_norm (2-norm of the homogeneous-system\n"
        "residual), converged and res_history (per-iteration residual norms)."
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
        "Matrix-free Ginkgo solve of the multi-level COMPOSITE MLLinOp system L(sol) = rhs\n"
        "on a 2+ level AMR hierarchy (one sol/rhs MultiFab per level, coarsest first).\n\n"
        "The mat-vec is the multi-level MLMG::apply, so coarse/fine interpolation, reflux\n"
        "and covered-cell average_down are AMReX's and the solved system is identical to\n"
        "MLMG's own composite solve. Covered coarse cells are slaved, not DOFs: their rhs\n"
        "entries are replaced internally by the average_down of the fine rhs, and on return\n"
        "they hold the average_down of the fine solution. sol's incoming values are the\n"
        "initial guess (residual-correction form, set_level_bc honored). `sign` as in\n"
        "ginkgo_solve: -1.0 for MLPoisson, +1.0 for MLABecLaplacian. The composite operator\n"
        "is not exactly symmetric (c/f interpolation vs reflux), so solver='bicgstab'\n"
        "(default) or 'gmres' are safe; 'cg' may work in practice. Stops when\n"
        "||r_k||_2 <= rtol*||rhs||_2 in the composite norm (or <= atol when atol > 0).\n"
        "`executor` is a NeoN SerialExecutor / CPUExecutor / GPUExecutor.\n\n"
        "Returns a dict with num_iters, res_norm, converged and res_history."
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
        "The matrix is carried as OpenFOAM-style AMReX fields: alpha is the cell-centred\n"
        "diagonal SOURCE (ddt/Sp/reaction), and u{x,y,z}/l{x,y,z} are the face-centred\n"
        "upper/lower off-diagonal coefficients (pass the same field for u* and l* for a\n"
        "symmetric matrix). The full diagonal is alpha - negSumDiag(faces), computed once\n"
        "when the operator is built. `solver` is 'cg' (SPD only), 'bicgstab' (default) or\n"
        "'gmres'; sol's incoming values seed the initial guess, and all three stop when\n"
        "||r_k||_2 <= rtol*||rhs||_2. Returns a dict with num_iters and res_norm."
    );

    // Persistent solver: operator + Ginkgo solver built once, solved many times.
    // FaceCoeffSolver recomputes the mat-vec from the face coefficients each apply.
    bindPersistent<FaceCoeffSolver>(m, "FaceCoeffSolver");

    m.def(
        "_la_matrix_solve",
        [](MultiFab& alpha,
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
            auto matrix = makeLaMatrix(symmetry, nexec, alpha, geom, bcArr);
            if (assemble_before_write)
            {
                // Force a toLinOp() over the still-zero coefficients: an operator cached
                // across the write below would then solve with the zero matrix.
                auto stale = toLinOp(matrix);
                (void)stale;
            }
            writeCoefficients(matrix, alpha, ux, lx, uy, ly, uz, lz);
            // A COPY shares the coefficient fields with the original (they sit behind
            // shared_ptr), so it must land on identical bits.
            blockamr::la::MFFaceCoeffs work = via_copy ? matrix : std::move(matrix);
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
            d["local_rows"] = work.localRows();
            d["symmetric"] = work.symmetry() == blockamr::la::Symmetry::symmetric;
            return d;
        },
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
        "Solve A sol = rhs through a blockamr::la::MFFaceCoeffs (matrix-free).\n\n"
        "The matrix allocates its OWN coefficient fields; the seven passed here are copied\n"
        "into them. With symmetry='symmetric' there is no `lower` and l* are ignored.\n"
        "via_copy=True solves through a COPY of the matrix, which SHARES those fields;\n"
        "assemble_before_write=True calls la::toLinOp once before the coefficients are\n"
        "written, which must not freeze the operator.\n\n"
        "Returns the usual solve dict plus local_rows and symmetric."
    );
    m.def(
        "_la_matrix_probe",
        [](MultiFab& alpha,
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
            auto matrix = makeLaMatrix(symmetry, nexec, alpha, geom, bcArr);

            nb::dict d;
            d["symmetric"] = matrix.symmetry() == blockamr::la::Symmetry::symmetric;
            d["local_rows"] = matrix.localRows();
            {
                d["lower_empty"] = !matrix.lower.has_value();
                // Structurally false: Cell/FaceFieldLevel have no empty state, so the
                // invariant is asserted at compile time (faceCoeffMatrix.hpp) instead.
                d["upper_empty"] = false;
                d["diag_empty"] = false;
                // The ABSENCE of `lower` is what "symmetric" means to a coefficient reader;
                // symmetry() derives from the same storage.
                d["reports_symmetric"] = !matrix.lower.has_value();
            }
            writeCoefficients(matrix, alpha, ux, lx, uy, ly, uz, lz);

            // Every toLinOp() below is held alive to the end of the scope, so two distinct
            // operators can never land on the same address.
            auto op1 = toLinOp(matrix);
            auto op2 = toLinOp(matrix);
            d["op_rows"] = static_cast<std::size_t>(op1->get_size()[0]);
            d["op_stable_without_write"] = (op1.get() == op2.get());
            // A REAL write to the diagonal source, not merely acquiring a handle: with the
            // coefficient fields public there is no "handle taken" moment left to observe,
            // so the probe writes what a caller would write.
            writeField(*matrix.alpha, alpha, "alpha");
            auto op3 = toLinOp(matrix);
            d["op_rebuilt_after_coefficients"] = (op3.get() != op2.get());
            matrix.zero();
            auto op4 = toLinOp(matrix);
            d["op_rebuilt_after_zero"] = (op4.get() != op3.get());
            return d;
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
        nb::arg("symmetry") = "symmetric",
        nb::arg("bc") = std::vector<std::string>(6, "periodic"),
        "Introspect a blockamr::la::MFFaceCoeffs without solving: symmetry, localRows,\n"
        "whether `lower` is absent, the row count la::toLinOp hands back, and whether that\n"
        "operator is rebuilt after a coefficient write / zero() or reused."
    );

    m.def(
        "_la_system_solve",
        [](MultiFab& gamma,
           MultiFab& alpha,
           const Geometry& geom,
           MultiFab& rhs,
           MultiFab& sol,
           MultiFab& alpha_out,
           MultiFab& ux_out,
           MultiFab& uy_out,
           MultiFab& uz_out,
           MultiFab* lx_out,
           MultiFab* ly_out,
           MultiFab* lz_out,
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
            auto matrix = makeLaMatrix(symmetry, nexec, gamma, geom, bcArr);
            writeDiagSource(matrix, alpha);

            blockamr::la::LinearSystem system(matrix, rhs);
            system += blockamr::ops::Laplacian(gamma, bc_data);
            readCoefficients(matrix, alpha_out, ux_out, uy_out, uz_out, lx_out, ly_out, lz_out);

            SolverConfig cfg;
            cfg.solver = solver;
            cfg.solverKind = parseSolverKind(solver);
            cfg.maxIter = max_iter;
            cfg.rtol = rtol;
            cfg.atol = atol;
            cfg.projectNullspace = project_nullspace;

            blockamr::la::Solver s(nexec, cfg);
            nb::dict d = toDict(s.solve(system, sol));
            d["local_rows"] = system.localRows();
            d["symmetric"] = system.matrix().symmetry() == blockamr::la::Symmetry::symmetric;
            // Non-owning: the rhs the solve reads IS the caller's MultiFab.
            d["rhs_aliases_input"] = (&system.rhs() == &rhs);
            return d;
        },
        nb::arg("gamma"),
        nb::arg("alpha"),
        nb::arg("geom"),
        nb::arg("rhs"),
        nb::arg("sol"),
        nb::arg("alpha_out"),
        nb::arg("ux_out"),
        nb::arg("uy_out"),
        nb::arg("uz_out"),
        nb::arg("lx_out").none() = nb::none(),
        nb::arg("ly_out").none() = nb::none(),
        nb::arg("lz_out").none() = nb::none(),
        nb::arg("executor") = nb::none(),
        nb::arg("symmetry") = "symmetric",
        nb::arg("solver") = "cg",
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        nb::arg("atol") = 0.0,
        nb::arg("project_nullspace") = false,
        nb::arg("bc") = std::vector<std::string>(6, "periodic"),
        nb::arg("bc_data").none() = nb::none(),
        "Assemble A through `system += ops::Laplacian(gamma)` and solve\n"
        "A sol = rhs with blockamr::la::Solver.\n\n"
        "`alpha` is the cell-centred diagonal SOURCE and is written into the matrix's own\n"
        "field directly -- there is no ops::Ddt yet. The FACE coefficients\n"
        "are the operator's alone, as is the BC fold: a non-periodic side gets a zero face\n"
        "coefficient and (sign-1)*aF on alpha, and with bc_data set also -aF*scale*g on the\n"
        "rhs -- which MUTATES the rhs passed in, since LinearSystem is non-owning. The\n"
        "assembled coefficients are copied back into alpha_out/u{x,y,z}_out for a bitwise\n"
        "comparison; the optional l{x,y,z}_out receive the LOW side and are written only when\n"
        "the matrix has a `lower`, i.e. with symmetry='asymmetric'.\n\n"
        "Returns the usual solve dict plus local_rows, symmetric and rhs_aliases_input."
    );
    m.def(
        "_la_system_probe",
        [](MultiFab& gamma,
           MultiFab& alpha,
           const Geometry& geom,
           MultiFab& rhs,
           MultiFab& alpha_out,
           MultiFab& ux_out,
           MultiFab& uy_out,
           MultiFab& uz_out,
           MultiFab* lx_out,
           MultiFab* ly_out,
           MultiFab* lz_out,
           std::optional<NeoN::Executor> executor,
           const std::string& symmetry,
           const std::vector<std::string>& bc,
           int n_apply,
           bool zero_after,
           const MultiFab* bc_data)
        {
            const NeoN::Executor nexec = executor.value_or(NeoN::createDefaultExecutor());
            const BcArray bcArr = parseBc(bc, geom, "_la_system_probe");
            auto matrix = makeLaMatrix(symmetry, nexec, gamma, geom, bcArr);
            writeDiagSource(matrix, alpha);

            blockamr::la::LinearSystem system(matrix, rhs);
            for (int i = 0; i < n_apply; ++i)
            {
                system += blockamr::ops::Laplacian(gamma, bc_data);
            }
            if (zero_after)
            {
                system.zero();
            }
            readCoefficients(matrix, alpha_out, ux_out, uy_out, uz_out, lx_out, ly_out, lz_out);

            nb::dict d;
            d["local_rows"] = system.localRows();
            d["symmetric"] = system.matrix().symmetry() == blockamr::la::Symmetry::symmetric;
            d["rhs_aliases_input"] = (&system.rhs() == &rhs);
            d["rhs_sum"] = system.rhs().sum(0);
            return d;
        },
        nb::arg("gamma"),
        nb::arg("alpha"),
        nb::arg("geom"),
        nb::arg("rhs"),
        nb::arg("alpha_out"),
        nb::arg("ux_out"),
        nb::arg("uy_out"),
        nb::arg("uz_out"),
        nb::arg("lx_out").none() = nb::none(),
        nb::arg("ly_out").none() = nb::none(),
        nb::arg("lz_out").none() = nb::none(),
        nb::arg("executor") = nb::none(),
        nb::arg("symmetry") = "symmetric",
        nb::arg("bc") = std::vector<std::string>(6, "periodic"),
        nb::arg("n_apply") = 1,
        nb::arg("zero_after") = false,
        nb::arg("bc_data").none() = nb::none(),
        "Assemble a blockamr::la::LinearSystem without solving it.\n\n"
        "Applies ops::Laplacian `n_apply` times (operators ACCUMULATE, so twice is twice the\n"
        "coefficients) and optionally calls LinearSystem::zero() afterwards, which clears the\n"
        "coefficients AND the rhs. Copies the result into alpha_out/u{x,y,z}_out -- and into\n"
        "the optional l{x,y,z}_out, but only when the matrix has a `lower`\n"
        "(symmetry='asymmetric') -- and reports local_rows, symmetric, rhs_aliases_input\n"
        "and the rhs sum."
    );

    // The fine-level diagonal computeFaceCoeffDiag derives from an MFFaceCoeffs' own
    // coefficients: the diagonal is deliberately NOT stored on the matrix at all.
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
            const blockamr::MeshLevel mesh {alpha.boxArray(), alpha.DistributionMap(), geom};
            if (symmetry != "symmetric" && symmetry != "asymmetric")
            {
                throw std::runtime_error("la matrix: unknown symmetry '" + symmetry + "'");
            }
            auto m = (symmetry == "symmetric")
                       ? blockamr::la::MFFaceCoeffs::symmetric(nexec, mesh, bcArr)
                       : blockamr::la::MFFaceCoeffs::asymmetric(nexec, mesh, bcArr);
            {
                writeField(*m.alpha, alpha, "alpha");
                writeField(m.upper[0], ux, "ux");
                writeField(m.upper[1], uy, "uy");
                writeField(m.upper[2], uz, "uz");
                if (m.lower.has_value())
                {
                    writeField((*m.lower)[0], lx, "lx");
                    writeField((*m.lower)[1], ly, "ly");
                    writeField((*m.lower)[2], lz, "lz");
                }
            }
            // The format stores no diagonal, so the probe derives one the way any consumer
            // does: computeFaceCoeffDiag over the matrix's own coefficients, into scratch.
            auto copyOutDiagonal = [&m](MultiFab& out, const char* what)
            {
                MultiFab diag(m.mesh.ba, m.mesh.dm, 1, 0);
                computeFaceCoeffDiag(
                    m.exec,
                    blockamr::CellFieldLevel {blockamr::nonOwning(diag)},
                    m.alpha,
                    m.upper,
                    m.storedLower()
                );
                writeField(out, diag, what);
            };
            copyOutDiagonal(diag_out, "diag_out");

            if (alpha2 != nullptr)
            {
                if (diag2_out == nullptr)
                {
                    throw std::runtime_error("_la_stored_diagonal: alpha2 needs diag2_out");
                }
                // A copy shares the coefficient fields, so a write through it must be visible
                // in the diagonal derived from the ORIGINAL.
                if (rewrite_through_copy)
                {
                    blockamr::la::MFFaceCoeffs copy = m;
                    writeField(*copy.alpha, *alpha2, "alpha2");
                }
                else
                {
                    writeField(*m.alpha, *alpha2, "alpha2");
                }
                copyOutDiagonal(*diag2_out, "diag2_out");
            }
            if (diag_zero_out != nullptr)
            {
                m.zero();
                copyOutDiagonal(*diag_zero_out, "diag_zero_out");
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
        "Copy out an MFFaceCoeffs' fine-level diagonal, alpha - sum(faces).\n\n"
        "Writes the seven coefficients through the format's own field handles and copies\n"
        "the diagonal derived from them into diag_out. With alpha2/diag2_out, rewrites the\n"
        "diagonal SOURCE afterwards and copies the diagonal again (rewrite_through_copy routes\n"
        "that write through a COPY of the format, which shares the coefficient fields). With\n"
        "diag_zero_out, calls zero() last and copies the diagonal that follows."
    );

    // blockamr::la as real Python classes, over the same machinery the underscore-prefixed
    // seams above drive. `blockamr.linear_algebra` re-exports them under their public names.

    // The layout triple as one object (core/meshLevel.hpp). Constructor only -- nothing reads
    // a MeshLevel's members back from Python.
    nb::class_<blockamr::MeshLevel>(m, "MeshLevel")
        .def(
            "__init__",
            [](blockamr::MeshLevel* self,
               const BoxArray& ba,
               const DistributionMapping& dm,
               const Geometry& geom) {
                new (self) blockamr::MeshLevel {ba, dm, geom};
            },
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("geom"),
            "One AMR level's layout: a BoxArray, its DistributionMapping and the Geometry over\n"
            "it. Held by value -- ba/dm are refcounted handles, so a copy shares the layout."
        );

    // One binding per factory rather than one taking a format string: there is one matrix,
    // and its symmetry is chosen once at allocation and never dispatched on again.
    const auto matrixFactory = [](auto make)
    {
        return [make](
                   const blockamr::MeshLevel& mesh,
                   std::optional<NeoN::Executor> executor,
                   const std::vector<std::string>& bc
               )
        {
            const NeoN::Executor nexec = executor.value_or(NeoN::createDefaultExecutor());
            return make(nexec, mesh, parseBc(bc, mesh.geom, "la matrix"));
        };
    };
    nb::class_<blockamr::la::MFFaceCoeffs>(m, "MFFaceCoeffs")
        .def_static(
            "symmetric",
            matrixFactory([](auto&&... a) { return blockamr::la::MFFaceCoeffs::symmetric(a...); }),
            nb::arg("mesh"),
            nb::arg("executor") = nb::none(),
            nb::arg("bc") = std::vector<std::string>(6, "periodic")
        )
        .def_static(
            "asymmetric",
            matrixFactory([](auto&&... a) { return blockamr::la::MFFaceCoeffs::asymmetric(a...); }),
            nb::arg("mesh"),
            nb::arg("executor") = nb::none(),
            nb::arg("bc") = std::vector<std::string>(6, "periodic")
        )
        .def("local_rows", &blockamr::la::MFFaceCoeffs::localRows)
        .def(
            "is_symmetric",
            [](const blockamr::la::MFFaceCoeffs& mat)
            { return mat.symmetry() == blockamr::la::Symmetry::symmetric; }
        )
        .def("zero", &blockamr::la::MFFaceCoeffs::zero)
        .def(
            "diagonal_source",
            [](blockamr::la::MFFaceCoeffs& mat, const MultiFab& alpha)
            { writeDiagSource(mat, alpha); },
            nb::arg("alpha"),
            "Write the cell-centred diagonal SOURCE (ddt/Sp/reaction) straight into the\n"
            "matrix's `alpha`. There is no ops::Ddt yet, so this is the only way to set\n"
            "it; it is NOT the matrix diagonal, which stays alpha - sum(faces)."
        );

    // Opaque on purpose: an operator's whole surface is `assemble`, which takes the system --
    // so `system += op` is all a caller can do with one, here as in C++.
    nb::class_<blockamr::ops::Laplacian>(m, "Laplacian");

    m.def(
        "la_laplacian",
        [](const MultiFab& gamma, const MultiFab* bcData)
        { return blockamr::ops::Laplacian(gamma, bcData); },
        nb::arg("gamma"),
        nb::arg("bc_data").none() = nb::none(),
        // gamma and bc_data are held BY POINTER (laplacian.hpp, LIFETIME) and
        // read when `system +=` runs, which may be much later.
        nb::keep_alive<0, 1>(),
        nb::keep_alive<0, 2>(),
        "The implicit diffusion term as face coefficients: `system += laplacian(...)`.\n\n"
        "The mesh and the domain BCs are read off the system's MATRIX, so neither is an\n"
        "argument here. Named `la_laplacian` because `blockamr.laplacian` is already the\n"
        "stencil-kernel binding; `blockamr.linear_algebra.laplacian` is the name callers use."
    );

    nb::class_<blockamr::la::LinearSystem>(m, "LinearSystem")
        .def(
            nb::init<blockamr::la::MFFaceCoeffs&, MultiFab&>(),
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
               const blockamr::ops::Laplacian& op) -> blockamr::la::LinearSystem&
            {
                s += op;
                return s;
            },
            nb::arg("op"),
            nb::rv_policy::none
        )
        .def("zero", &blockamr::la::LinearSystem::zero)
        .def("local_rows", &blockamr::la::LinearSystem::localRows)
        .def(
            "matrix",
            nb::overload_cast<>(&blockamr::la::LinearSystem::matrix),
            nb::rv_policy::reference_internal
        )
        .def(
            "rhs",
            nb::overload_cast<>(&blockamr::la::LinearSystem::rhs),
            nb::rv_policy::reference_internal
        );

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
               const std::string& norm,
               int precond_cycles,
               int gmg_pre_sweeps,
               int gmg_post_sweeps,
               int gmg_coarsest_sweeps,
               int gmg_max_levels,
               int gmg_min_bottom,
               const std::string& gmg_smoother,
               const std::string& gmg_precision,
               const std::string& gmg_coeff_precision,
               double gmg_omega)
            {
                // The same builder FaceCoeffSolver goes through, so the two Python surfaces
                // cannot describe different cycles. The arguments this class does not model --
                // the MLMG preconditioner, bc/bc_data, the mixed-precision inner tolerances,
                // and DELIBERATELY the five knobs gmg_agg_l0_size/symmetric/gmg_bottom_* -- are
                // passed as their C++ defaults, so they stay unreachable here rather than
                // becoming a new public knob, and are refused rather than accepted and ignored.
                new (self) PyLaSolver {parseSolverConfig(
                    solver,
                    max_iter,
                    rtol,
                    atol,
                    project_nullspace,
                    configDefaults().precondMlmg,
                    precond_cycles,
                    configDefaults().bc,
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
                    configDefaults().gmg.aggLevel0Size,
                    configDefaults().gmg.symmetric,
                    configDefaults().gmg.bottomSolver,
                    configDefaults().gmg.bottomMaxIter,
                    configDefaults().gmg.bottomRtol,
                    configDefaults().mpInnerRtol,
                    configDefaults().mpInnerMaxIter,
                    norm,
                    configDefaults().bcData
                )};
            },
            nb::arg("solver") = configDefaults().solver,
            nb::arg("max_iter") = configDefaults().maxIter,
            nb::arg("rtol") = configDefaults().rtol,
            nb::arg("atol") = configDefaults().atol,
            nb::arg("project_nullspace") = configDefaults().projectNullspace,
            nb::arg("precond") = configDefaults().precond,
            nb::arg("norm") = configDefaults().norm,
            nb::arg("precond_cycles") = configDefaults().precondCycles,
            nb::arg("gmg_pre_sweeps") = configDefaults().gmg.preSweeps,
            nb::arg("gmg_post_sweeps") = configDefaults().gmg.postSweeps,
            nb::arg("gmg_coarsest_sweeps") = configDefaults().gmg.coarsestSweeps,
            nb::arg("gmg_max_levels") = configDefaults().gmg.maxLevels,
            nb::arg("gmg_min_bottom") = configDefaults().gmg.minBottom,
            nb::arg("gmg_smoother") = configDefaults().gmg.smoother,
            nb::arg("gmg_precision") = configDefaults().gmg.precision,
            nb::arg("gmg_coeff_precision") = configDefaults().gmg.coeffPrecision,
            nb::arg("gmg_omega") = configDefaults().gmg.omega
        )
        .def(
            "solve",
            [](const PyLaSolver& s, const blockamr::la::LinearSystem& system, MultiFab& sol)
            { return toDict(s.solve(system, sol)); },
            nb::arg("system"),
            nb::arg("sol"),
            "Solve `system` into `sol` (in place; its incoming values seed the initial\n"
            "guess). The executor comes from the system's MATRIX.\n\n"
            "The preconditioner is built from the coefficients the MATRIX holds, so\n"
            "precond='gmg'/'gmg_kokkos'/'mlmg' all work here.\n\n"
            "solver in ('gmg', 'ir', 'mpir') also RAISES: those want the hierarchy as the\n"
            "SOLVER rather than as a preconditioner. Use blockamr.FaceCoeffSolver for them."
        );

    // Profiling accessors (namespace prof). Empty unless BLOCKAMR_PROFILE=1.
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
