// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>

#include "NeoN/blockAmr/linearAlgebra/faceCoeffMatrix.hpp"
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"

// The ONE place linearAlgebra/ hands a gko:: type to a caller. Free functions rather than
// members of MFFaceCoeffs, so the matrix -- and every operator that writes into it -- stays
// AMReX-only: an outward-facing adapter is not part of the thing it adapts.

namespace blockamr::la
{

/* @brief The matrix-free operator over `matrix`'s coefficients.
 *
 * Built fresh per call, not cached: the operator stages PINNED COPIES of the coefficient
 * fields on the host path, so a cached one would go stale after a write to them on that
 * path. This is a per-solve call, not a per-iteration one.
 */
std::shared_ptr<const gko::LinOp> toLinOp(const MFFaceCoeffs& matrix);

/* @brief The preconditioner for `config`, from THIS matrix's own coefficients:
 *        none / gmg / gmg_kokkos / mlmg, never declined.
 */
std::shared_ptr<const gko::LinOp>
makeHierarchy(const MFFaceCoeffs& matrix, const SolverConfig& config);

} // namespace blockamr::la
