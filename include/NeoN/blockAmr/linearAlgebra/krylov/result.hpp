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

// What every solve entry point returns. Its key set is a CONTRACT, pinned by
// test_gmg_solver_stats_keys_match_cg, so every path fills contraction/diagnostic. The
// optionals are for ginkgo_solve_face_coeffs, whose dict carries only num_iters/res_norm.
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
