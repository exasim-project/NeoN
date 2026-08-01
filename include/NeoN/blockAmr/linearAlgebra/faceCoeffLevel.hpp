// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/meshLevel.hpp"

// Its own header rather than precond.hpp's, because the operator and the preconditioner both
// take one: a low-level header including the factory header just to name this bundle inverts
// the layering, and it is the factories that build on the bundle, not the other way round.

namespace blockamr::la
{

/* @brief The coefficient fields one level of the face-coefficient operator is built from, as
 *        the grouped handles rather than seven loose MultiFabs: `lower` is the STORED low side
 *        (aliasing upper when symmetric), and `mesh` carries the ba/dm/geom the hierarchy
 *        coarsens from (core/fieldLevel.hpp). Grouped because seven adjacent MultiFab of
 *        identical type is a transposition hazard, and la::FaceCoeffs cannot express it --
 *        that one is FabArray<BaseFab<T>>-level, and const.
 */
struct FaceCoeffLevel
{
    CellFieldLevel alpha;
    FaceFieldLevel upper;
    FaceFieldLevel lower;
    MeshLevel mesh;
};

} // namespace blockamr::la
