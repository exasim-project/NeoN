// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>

namespace blockamr
{

/* @brief The AMReX layout triple -- BoxArray, DistributionMapping, Geometry --
 *        travelling as ONE argument.
 *
 * The `Level` suffix matches CellFieldLevel/FaceFieldLevel (core/fieldLevel.hpp):
 * this is the layout of ONE AMR level, which is the granularity everything in the
 * linear algebra works at. Python's `blockamr.Mesh`/`AmrMesh` (python/blockamr/
 * mesh.py) are the multi-level containers and the name is already taken there.
 *
 * HELD BY VALUE, deliberately, and that is what makes it safe as a MEMBER as well
 * as a parameter: amrex::BoxArray and amrex::DistributionMapping are refcounted
 * handles (a copy is a refcount bump onto shared, immutable layout data) and
 * amrex::Geometry is a plain value that every consumer in this component already
 * stores by value. So a copy is cheap and a member cannot dangle. An earlier
 * design held ba/dm by POINTER, which forced the rule "parameter type only, never
 * a member"; value semantics retires that rule rather than working around it.
 *
 * It is a plain aggregate on purpose. There are no accessors because no caller in
 * the coefficient path spells one through a MeshLevel: the two places that reach
 * for CellSize/periodicity/Domain (operators/laplacian.cpp,
 * linearAlgebra/matrixFree/faceCoeffOp.cpp) read them off a `geom` MEMBER they
 * already hold, not off a MeshLevel parameter. Add dx()/periodicity()/fillHalo()
 * when a caller wants them, not before.
 *
 * The executor is deliberately NOT here. la routes it through the MATRIX
 * (linearAlgebra/coefficients.hpp, IsMatrix::executor()) so an operator launches
 * where the coefficient fields live; a second source for it is exactly the
 * failure this grouping exists to prevent.
 */
struct MeshLevel
{
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    amrex::Geometry geom;
};

} // namespace blockamr
