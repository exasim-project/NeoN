// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "../common/types.hpp" // MLMG alias

namespace blockamr::solvers
{

// Native-GMG hierarchy knobs (precond="gmg"/"gmg_kokkos", solver="gmg"/"ir"/"mpir").
// Every field name/default matches the historical nb::arg exactly; strings stay
// strings (T10 introduces enums and centralises the dispatch built on them — this
// struct only stops the 22-parameter buildGmgHierarchy signature from being typed
// out four times).
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
// nanobind boundary, ginkgo_solve.cpp) from the 27 raw __init__ arguments;
// both persistent-solver constructors take `const SolverConfig&` instead of
// spelling out all 27 (36 minus the 9 fixed: executor, geom, alpha..lz).
struct SolverConfig
{
    std::string solver = "bicgstab";
    int maxIter = 1000;
    double rtol = 1e-10;
    double atol = 0.0;
    bool projectNullspace = false;
    MLMG* precondMlmg = nullptr;
    int precondCycles = 1;
    std::vector<std::string> bc = std::vector<std::string>(6, "periodic");
    std::string precond = "none";
    GmgConfig gmg;
    double mpInnerRtol = 1e-2;
    int mpInnerMaxIter = 20;
    std::string norm = "l2";
    const amrex::MultiFab* bcData = nullptr;
};

} // namespace blockamr::solvers
