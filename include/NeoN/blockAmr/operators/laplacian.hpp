// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/linearAlgebra/coefficients.hpp"

namespace blockamr::ops
{

/* @class Laplacian
 * @brief The implicit diffusion discretisation, written as face coefficients.
 *
 * No base class: satisfy la::IsOperator and you are an operator. It is reached
 * only through `system += ops::Laplacian(gamma, geom, bc)` -- la::Coefficients,
 * the argument of `assemble`, has a private constructor whose only friend is
 * la::LinearSystem, so a caller cannot produce one.
 *
 * WHAT IT WRITES, exactly:
 *
 *   upper[d](face) += -gammaFace / dx[d]^2       (and lower[d] when asymmetric)
 *
 * with `gammaFace` the arithmetic mean of the two cells the face separates --
 * EXCEPT on a domain face of a non-periodic side, where the boundary condition is
 * folded in instead (below).
 *
 * THE BOUNDARY FOLD (S6b). core/bc.hpp fills the ghost as
 * `ghost = sign*interior + scale*g`, with (sign -1, scale 2, g = u on the FACE)
 * for Dirichlet and (sign +1, scale dx[d], g = du/dn outward) for Neumann. A
 * boundary face with coefficient aF therefore contributes
 * `aF*(sign*pC + scale*g)` to its cell's row, which has no off-diagonal part at
 * all. So this operator writes, for that face and its cell C:
 *
 *   aF      ->  0                    (nothing is accumulated onto the face)
 *   diag(C) += (sign - 1) * aF       Dirichlet: -= 2*aF ; Neumann: unchanged
 *   rhs(C)  -= aF * scale * g        (bcData only; g read from ITS ghost cell)
 *
 * `(sign - 1)` and not `sign` because `diag` here is still `alpha`, the diagonal
 * SOURCE, and the matrix diagonal `alpha - sum(faces)` is derived from it
 * (faceCoeffMatrix.hpp). Not accumulating aF already removes `-aF` from that sum,
 * so alpha only owes the remaining `sign*aF`:
 *
 *   alpha' - sum(faces)' = [alpha + (sign-1)aF] - [sum - aF]
 *                        = alpha - sum + sign*aF                              (*)
 *
 * which is exactly the row assembleFaceCoeffCsr's `side()` lambda builds for a
 * non-periodic side (sparse/csr.cpp, S6a) and exactly the row FaceCoeffOp's ghost
 * reflection produces. The three agree by construction.
 *
 * THE FORMATS STILL FOLD TOO, AND THAT IS FINE -- BUT ONLY BECAUSE aF IS ZEROED.
 * MFFaceCoeffs and CsrMatrix still hand their BcArray to FaceCoeffOp and
 * assembleFaceCoeffCsr (faceCoeffMatrix.hpp), which reflect the ghost / fold the
 * diagonal a second time. Every one of those folds is MULTIPLICATIVE in the face
 * coefficient -- `aF*(sign*pC)` in the stencil, `diag += sign*aFace` in csr.cpp's
 * side() -- so on a coefficient this operator folded they contribute exactly
 * nothing. Keeping `bc` on the formats is what preserves S6a's variable row
 * length (an all-zero BcArray would make csr.cpp emit an explicit 0.0 at the
 * wraparound column) and what a hand-written non-periodic coefficient set still
 * needs.
 *
 * The dependency runs one way and it is load-bearing: fold onto the diagonal
 * while leaving aF on the face and every non-periodic boundary is wrong, because
 * the format's fold then lands on a live coefficient. laplacian.cpp marks the
 * line that must not go, and names the probe that catches its removal.
 *
 * The legacy FaceCoeffSolver path is untouched and folds at apply time only.
 *
 * The stored diagonal is NOT written by hand. It is derived behind a dirty flag
 * that Matrix::coefficients() already set before this operator ran (S7), so
 * writing `diag` and the face fields is sufficient and touching it directly would
 * double-count.
 *
 * SIGN. The coefficient is NEGATIVE, so this operator contributes
 * `-div(gamma grad phi)` -- with the format's diagonal source `alpha`, a system
 * assembled as `alpha` plus one Laplacian is `alpha*phi - div(gamma grad phi)`.
 * That is the sign every existing blockAmr caller writes by hand (`-1/dx^2` on
 * every face, throughout test/ and bench/), and reproducing those callers exactly
 * is what this slice is for. OpenFOAM's `fvm::laplacian(gamma, phi)` is the other
 * sign, i.e. this is `-fvm::laplacian`; there is no sign argument because the
 * constructor the design pins has none.
 *
 * LIFETIME. `gamma` is held by pointer and must outlive every `+=` this object is
 * used in. amrex::FabArray's copy constructor is deleted, so an operator cannot
 * own a field by value; and it should not, since the whole point of assembling
 * from a live field is that the caller's values are the ones read.
 */
class Laplacian
{
public:

    // `bcData` carries the INHOMOGENEOUS boundary datum, cell-centred on the
    // matrix's BoxArray with >= 1 ghost and the datum living in the ghost layer
    // (MLMG's setLevelBC contract, the same carrier FaceCoeffSolver's `bc_data`
    // takes). null means homogeneous; then no rhs is written at all.
    Laplacian(
        const amrex::MultiFab& gamma,
        amrex::Geometry geom,
        la::BcArray bc,
        const amrex::MultiFab* bcData = nullptr
    );

    // Accumulate, never assign -- several operators may share one system.
    void assemble(la::Coefficients c) const;

private:

    const amrex::MultiFab* gamma_;
    amrex::Geometry geom_;
    la::BcArray bc_;
    const amrex::MultiFab* bcData_;
};

static_assert(la::IsOperator<Laplacian>);

} // namespace blockamr::ops
