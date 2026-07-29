// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace blockamr::la
{

// What every solve entry point returns. The key set is a CONTRACT: a caller reads
// one dict without branching on which solver produced it (see
// test_gmg_solver_stats_keys_match_cg), so every path fills contraction/diagnostic.
//
// The std::optionals are for the ONE caller that never held that contract
// (ginkgo_solve_face_coeffs, whose historical dict carries only num_iters/res_norm):
// the nanobind converter in ginkgoSolve.cpp emits a key only for a set field, so that
// caller's Python-visible surface stays exactly what it was.
struct SolveResult
{
    std::int64_t num_iters = 0;
    double res_norm = 0.0;
    std::optional<bool> converged;
    std::optional<std::vector<double>> res_history;
    std::optional<double> contraction;
    std::optional<std::string> diagnostic;
};

} // namespace blockamr::la
