// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace blockamr::solvers
{

// Assemble the {num_iters, res_norm, converged, res_history, contraction,
// diagnostic} result dict returned by every solve entry point (the epilogue
// duplicated at several ginkgo_solve.cpp call sites). `res_history` is built
// from `resLogger.history()`.
//
// `contraction` and `diagnostic` are filled for EVERY path, not just the
// stationary V-cycle that needs them, because the key set is a contract: a
// caller reads one dict without branching on which solver produced it (see
// test_gmg_solver_stats_keys_match_cg). `diagnostic` is left empty here and
// only the stationary path fills it in -- its thresholds are calibrated for a
// V-cycle's roughly constant contraction and say nothing useful about a Krylov
// method, whose rate varies over the run.
//
// converged/res_history/contraction/diagnostic are std::optional because ONE
// caller (ginkgo_solve_face_coeffs) never held this contract -- its historical
// dict carries only num_iters/res_norm -- and the nanobind converter in
// ginkgo_solve.cpp emits a key only when the corresponding field is set, so
// that caller's Python-visible surface stays exactly what it was.
struct SolveResult
{
    std::int64_t num_iters = 0;
    double res_norm = 0.0;
    std::optional<bool> converged;
    std::optional<std::vector<double>> res_history;
    std::optional<double> contraction;
    std::optional<std::string> diagnostic;
};

} // namespace blockamr::solvers
