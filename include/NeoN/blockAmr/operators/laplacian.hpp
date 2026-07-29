// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MultiFab.H>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/linearAlgebra/coefficients.hpp"

namespace blockamr::ops
{

/* @class Laplacian
 * @brief The implicit diffusion discretisation, written as face coefficients.
 *
 * No base class: satisfy la::IsOperator and you are an operator. Reachable only
 * through `system += ops::Laplacian(gamma, bc)`, since la::Coefficients has a
 * private constructor whose only friend is la::LinearSystem.
 *
 * Writes `upper[d](face) += -gammaFace / dx[d]^2` (and lower[d] when asymmetric),
 * gammaFace being the mean of the two cells the face separates. The coefficient is
 * NEGATIVE, so with the format's diagonal source a system is
 * `alpha*phi - div(gamma grad phi)` -- the sign every existing blockAmr caller
 * writes by hand, i.e. this is `-fvm::laplacian`.
 *
 * BOUNDARY-CONDITION SPLIT. A non-periodic domain face carries its REAL
 * coefficient, with gamma taken from the boundary cell ITSELF -- the ghost beyond
 * it is never filled, so reading it would read recycled arena memory. The DIAGONAL
 * half of the homogeneous BC (core/bc.hpp fills `ghost = sign*interior + scale*g`:
 * Dirichlet sign -1 scale 2 with g = u on the FACE, Neumann sign +1 scale dx[d]
 * with g = du/dn outward) belongs to the CONSUMER, applied per level: FaceCoeffOp
 * reflects the ghost on every apply (matrixFree/faceCoeffOp.cpp),
 * assembleFaceCoeffCsr folds `diag += sign*aFace` (sparse/csr.cpp), and the GMG
 * hierarchy reflects on every level it builds (gmg/gmgPrecond.hpp). All three are
 * MULTIPLICATIVE in the face coefficient, so it must stay live -- which makes `bc`
 * on the formats load-bearing arithmetic; faceCoeffMatrix.hpp holds the other half.
 *
 * WHY THAT DIRECTION, when folding would give the same FINE matrix: the folded
 * `(sign-1)*aF` is dx-DEPENDENT (2*gamma/dx^2 for Dirichlet) yet sat in `alpha`,
 * where gmgRestrict coarsens it by a plain eight-child volume average correct only
 * for a dx-INDEPENDENT density, whereas on the face it coarsens by the correct 1/4
 * law (gmgCoarsenFace). Folded, every coarse level inherited a boundary diagonal
 * too strong and the V-cycle degraded as the mesh refined. DO NOT re-zero the face
 * coefficient to make room for an operator-side fold: such a fold can only ever be
 * right on the finest level.
 *
 * MEASURED (same rhs, tolerance and cycle shape): fully-periodic is 8 iterations
 * at 64/128/256^3 either way, solutions bitwise equal; fully-Dirichlet is 8/8/8
 * unfolded against 12/13/14 folded (1.69x/1.72x/1.74x slower); Neumann has
 * (sign-1) == 0, so nothing was ever folded and both conventions agree.
 *
 * TRIPWIRE: test_la_boundary_conditions.py::
 * test_laplacian_writes_the_boundary_face_coefficient -- load-bearing as the only
 * test that reaches NEUMANN, and because solve-level tests stay green either way.
 * test_the_two_formats_agree_through_the_laplacian catches nothing here: both
 * formats fold whatever they are handed identically.
 *
 * What IS written here is the INHOMOGENEOUS constant `rhs(C) -= aF * scale * g`
 * (bcData only, g read from ITS ghost cell), because no la:: consumer can produce
 * one: MFFaceCoeffs::op() hands FaceCoeffOp a null bcData, so
 * FaceCoeffOpT::applyBcOffset is unreachable from la::Solver, and
 * assembleFaceCoeffCsr takes no datum at all. One writer, so no double count. The
 * legacy FaceCoeffSolver path is untouched and folds at apply time only.
 *
 * The stored diagonal is not written by hand: it is derived behind a dirty flag
 * Matrix::coefficients() already set, so writing `diag` and the face fields
 * suffices and touching it directly would double-count.
 *
 * LIFETIME. `gamma` is held by pointer and must outlive every `+=` -- FabArray's
 * copy constructor is deleted, and reading the caller's live values is the point.
 */
class Laplacian
{
public:

    // `bcData` carries the INHOMOGENEOUS datum, cell-centred on the matrix's
    // BoxArray with >= 1 ghost and the datum in the ghost layer (MLMG's
    // setLevelBC contract). null means homogeneous, and then no rhs is written.
    //
    // NO geometry argument: it arrives on the coefficients as `c.mesh`, so the
    // geometry scaled by and the geometry the matrix was built on cannot disagree.
    Laplacian(
        const amrex::MultiFab& gamma, la::BcArray bc, const amrex::MultiFab* bcData = nullptr
    );

    // Accumulate, never assign -- several operators may share one system.
    void assemble(la::Coefficients c) const;

private:

    const amrex::MultiFab* gamma_;
    la::BcArray bc_;
    const amrex::MultiFab* bcData_;
};

static_assert(la::IsOperator<Laplacian>);

} // namespace blockamr::ops
