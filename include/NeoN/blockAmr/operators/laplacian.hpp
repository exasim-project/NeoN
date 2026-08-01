// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MultiFab.H>

#include "NeoN/blockAmr/linearAlgebra/linearSystem.hpp"

namespace blockamr::ops
{

/* @class Laplacian
 * @brief Implicit diffusion as face coefficients: `upper[d](face) += -gammaFace/dx[d]^2`
 *        (and lower[d] when asymmetric), so this is `-fvm::laplacian`. Reachable only
 *        through `system += ops::Laplacian(...)`. `gamma` is held by pointer (FabArray
 *        has no copy ctor) and must outlive every `+=`.
 */
class Laplacian
{
public:

    // `bcData` carries the INHOMOGENEOUS datum in its ghost layer (MLMG's setLevelBC
    // contract); null means homogeneous. Geometry and the domain BCs arrive on the matrix.
    explicit Laplacian(const amrex::MultiFab& gamma, const amrex::MultiFab* bcData = nullptr);

    // Accumulates; the signature IS the contract, no concept. Holds no geometry and no bc --
    // both are read off the system's matrix, so the operator has nowhere to keep a copy that
    // could disagree with it. A non-periodic domain face keeps its REAL coefficient -- do NOT
    // re-zero it to fold the BC here; the diagonal half is the consumer's, per level. Also
    // writes the inhomogeneous `rhs -= aF*scale*g`, which no la:: consumer can produce.
    // Tripwire:
    // test_la_boundary_conditions.py::test_laplacian_writes_the_boundary_face_coefficient. Why, and
    // the measurements: report/blockamr-linear-algebra-notes.md#laplacian-bcs
    void assemble(la::LinearSystem& sys) const;

private:

    const amrex::MultiFab* gamma_;
    const amrex::MultiFab* bcData_;
};

} // namespace blockamr::ops
