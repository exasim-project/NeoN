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

// Every value `solver` may spell, across both persistent-solver classes and every
// mode. Which of these a PARTICULAR class/mode accepts is a separate
// combination-legality question decided downstream against this enum (e.g.
// FaceCoeffCsrSolver never reaches gmg/ir/mpir); this only rejects an outright
// typo, once, at parseSolverConfig.
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

// Throws with buildKrylov's historical "unknown solver" wording. Parsing every
// spelling HERE (rather than in buildKrylov, which gmg/ir/mpir never reach) is
// what makes a typo among those three fail before any constructor work runs.
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

// Every value `precond` may spell. Per-class legality is decided downstream, as
// with SolverKind (FaceCoeffCsrSolver allows only none/mlmg, for instance).
enum class PrecondKind
{
    none,
    mlmg,
    gmg,
    gmg_kokkos
};

// Throws with FaceCoeffSolver's historical "unknown precond" wording, the more
// permissive of the two per-class messages. FaceCoeffCsrSolver's narrower check
// ("expected 'none' or 'mlmg'") still runs downstream on the parsed enum.
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
// INVARIANT: every field name/default matches the historical nb::arg exactly.
// smoother/precision/coeffPrecision/bottomSolver deliberately stay plain strings
// with no enum twin here: they are read only on a GMG path, so validating them
// unconditionally at parseSolverConfig would newly reject values that are
// silently inert today on a non-GMG path. They stay validated where they are read
// (gmgPrecond.hpp, gmgBottom.hpp, persistent.cpp, gmgKokkos/vcycle.hpp).
struct GmgConfig
{
    int preSweeps = 2;
    int postSweeps = 2;
    int coarsestSweeps = 16;
    int maxLevels = 0;
    int minBottom = 2;
    std::string smoother = "rbgs";
    std::string precision = "fp32";
    std::string coeffPrecision = "";
    double omega = 1.1;
    int aggLevel0Size = 0;
    bool symmetric = true;
    std::string bottomSolver = "smoother";
    int bottomMaxIter = 200;
    double bottomRtol = 1e-12;
};

// Every non-coefficient, non-geometry, non-executor constructor argument of
// FaceCoeffSolver/FaceCoeffCsrSolver, built once by parseSolverConfig at the
// nanobind boundary (ginkgoSolve.cpp) from the 27 raw __init__ arguments.
struct SolverConfig
{
    std::string solver = "bicgstab";
    // Parsed from `solver` once by parseSolverConfig and what every downstream
    // dispatch compares against. Kept ALONGSIDE `solver`, not instead of it:
    // several messages still interpolate the original spelling.
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
