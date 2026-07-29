// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "NeoN/blockAmr/core/types.hpp" // MLMG alias

namespace blockamr::la
{

// Every value `solver` may spell, across both persistent-solver classes and
// every mode (Krylov, native-GMG loop, Ginkgo-IR twin, mixed-precision IR).
// Which of these a PARTICULAR class/mode combination actually accepts is a
// separate, per-call-site combination-legality question (e.g.
// FaceCoeffCsrSolver never reaches SolverKind::gmg/ir/mpir; buildKrylov never
// sees gmg/mpir at all) decided downstream by comparing this enum — this only
// rejects an outright typo, once, at parseSolverConfig, rather than wherever
// the string first happens to be dispatched on.
enum class SolverKind
{
    cg,
    bicgstab,
    gmres,
    gcr,
    fcg,
    ir,
    gmg,
    mpir
};

// Throws with buildKrylov's historical "unknown solver" wording: this check
// used to live only there (reached for every spelling except the three —
// gmg/ir/mpir — the persistent-solver constructors intercept before ever
// calling it), so a typo among THOSE three was rejected late, after other
// constructor work had already run. Parsing every spelling here instead means
// the same typo is now rejected before any of it.
inline SolverKind parseSolverKind(const std::string& solver)
{
    if (solver == "cg") return SolverKind::cg;
    if (solver == "bicgstab") return SolverKind::bicgstab;
    if (solver == "gmres") return SolverKind::gmres;
    if (solver == "gcr") return SolverKind::gcr;
    if (solver == "fcg") return SolverKind::fcg;
    if (solver == "ir") return SolverKind::ir;
    if (solver == "gmg") return SolverKind::gmg;
    if (solver == "mpir") return SolverKind::mpir;
    throw std::runtime_error("ginkgo: unknown solver '" + solver + "'");
}

// Every value `precond` may spell. As with SolverKind, which of these a given
// solver class accepts is a per-class combination-legality question decided
// downstream (FaceCoeffCsrSolver only ever allows none/mlmg, for instance);
// this only rejects an outright typo.
enum class PrecondKind
{
    none,
    mlmg,
    gmg,
    gmg_kokkos
};

// Throws with FaceCoeffSolver's historical "unknown precond" wording (the
// more permissive of the two per-class messages, since it is the one that
// names all four spellings). FaceCoeffCsrSolver's narrower combination-
// legality check ("expected 'none' or 'mlmg'") is unaffected: it still runs,
// downstream, comparing this already-validated enum against its own accepted
// subset.
inline PrecondKind parsePrecondKind(const std::string& precond)
{
    if (precond == "none") return PrecondKind::none;
    if (precond == "mlmg") return PrecondKind::mlmg;
    if (precond == "gmg") return PrecondKind::gmg;
    if (precond == "gmg_kokkos") return PrecondKind::gmg_kokkos;
    throw std::runtime_error(
        "FaceCoeffSolver: unknown precond '" + precond
        + "' (expected 'none', 'mlmg', 'gmg' or 'gmg_kokkos')"
    );
}

// Native-GMG hierarchy knobs (precond="gmg"/"gmg_kokkos", solver="gmg"/"ir"/"mpir").
// Every field name/default matches the historical nb::arg exactly. smoother/
// precision/coeffPrecision/bottomSolver deliberately stay plain strings rather
// than gaining enum twins here: unlike solver/precond (validated on EVERY
// construction, unconditionally), these four are only ever read on a GMG-using
// path, so validating them unconditionally at parseSolverConfig would newly
// reject values that are silently inert today on a non-GMG path — a real,
// untested behaviour change T10's spec does not ask for. They stay validated
// where they are read (gmgPrecond.hpp, gmgBottom.hpp, persistent.cpp,
// gmgKokkos/vcycle.hpp's own Precision/parsePrecision), each already a single
// site.
struct GmgConfig
{
    int preSweeps = 2;
    int postSweeps = 2;
    int coarsestSweeps = 16;
    int maxLevels = 0;
    int minBottom = 2;
    std::string smoother = "rbgs";
    std::string precision = "fp64";
    std::string coeffPrecision = "";
    double omega = 1.1;
    int aggLevel0Size = 0;
    bool symmetric = true;
    std::string bottomSolver = "smoother";
    int bottomMaxIter = 200;
    double bottomRtol = 1e-12;
};

// Every non-coefficient, non-geometry, non-executor constructor argument of
// FaceCoeffSolver/FaceCoeffCsrSolver. Built once by parseSolverConfig (the
// nanobind boundary, ginkgoSolve.cpp) from the 27 raw __init__ arguments;
// both persistent-solver constructors take `const SolverConfig&` instead of
// spelling out all 27 (36 minus the 9 fixed: executor, geom, alpha..lz).
struct SolverConfig
{
    std::string solver = "bicgstab";
    // Parsed from `solver` once by parseSolverConfig; solverKind is what
    // every dispatch downstream now compares against instead of re-comparing
    // the string. Kept alongside `solver` (not instead of it) because several
    // messages still interpolate the original spelling.
    SolverKind solverKind = SolverKind::bicgstab;
    int maxIter = 1000;
    double rtol = 1e-10;
    double atol = 0.0;
    bool projectNullspace = false;
    MLMG* precondMlmg = nullptr;
    int precondCycles = 1;
    std::vector<std::string> bc = std::vector<std::string>(6, "periodic");
    std::string precond = "none";
    // Parsed from `precond` once by parseSolverConfig; see solverKind above.
    PrecondKind precondKind = PrecondKind::none;
    GmgConfig gmg;
    double mpInnerRtol = 1e-2;
    int mpInnerMaxIter = 20;
    std::string norm = "l2";
    const amrex::MultiFab* bcData = nullptr;
};

} // namespace blockamr::la
