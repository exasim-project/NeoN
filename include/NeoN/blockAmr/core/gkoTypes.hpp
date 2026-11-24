// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

// Split out of core/types.hpp so that solverConfig.hpp -- and with it the whole public
// SolverConfig surface -- reaches only the AMReX MLMG alias and no Ginkgo header.

namespace blockamr::la
{

using Dense = gko::matrix::Dense<double>;

} // namespace blockamr::la
