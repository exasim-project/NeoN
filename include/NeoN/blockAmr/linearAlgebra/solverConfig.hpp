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

// Every value `solver` may spell. Which a PARTICULAR class/mode accepts is decided
// downstream; this only rejects an outright typo, once, at parseSolverConfig.
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

// Throws with buildKrylov's historical "unknown solver" wording. Parsing HERE is what makes
// a typo in gmg/ir/mpir -- which never reach buildKrylov -- fail before any ctor work.
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

// Every value `precond` may spell; per-class legality is decided downstream.
enum class PrecondKind
{
    none,
    mlmg,
    gmg,
    gmg_kokkos
};

// Throws with FaceCoeffSolver's historical wording; per-class legality still runs downstream
// on the parsed enum.
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
// THE default list: the bindings' nb::arg defaults read these values rather than restate
// them. No enum twin for the string knobs: read only on a GMG path, so validating them here
// would newly reject values that are inert today. They stay validated where they are read.
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

    // Compiler-generated, so it covers a knob added above without anyone remembering to
    // extend it -- which lets a validator ask one question instead of comparing every field
    // by hand.
    friend bool operator==(const GmgConfig&, const GmgConfig&) = default;
};

/* @brief Everything GmgPrecondT's constructor reads that is neither executor, mesh,
 *        coefficients nor boundary condition. The cycle count is SEPARATE from GmgConfig
 *        because SolverConfig::precondCycles is shared with the MLMG preconditioner; the
 *        constructor should not have to know where in the config each half came from.
 */
struct GmgPrecondSpec
{
    int nCycles = 1;
    GmgConfig gmg {};
};

// Every non-coefficient, non-geometry, non-executor constructor argument of
// FaceCoeffSolver, built once by parseSolverConfig (bindings/linearAlgebra.cpp).
struct SolverConfig
{
    std::string solver = "bicgstab";
    // Parsed from `solver` once and what every downstream dispatch compares against. Kept
    // ALONGSIDE `solver`: several messages still interpolate the original spelling.
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
