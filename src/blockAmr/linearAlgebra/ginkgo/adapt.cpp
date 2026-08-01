// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/linearAlgebra/ginkgo/adapt.hpp"

#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.hpp"
#include "NeoN/blockAmr/linearAlgebra/precond.hpp"

namespace blockamr::la
{

namespace
{

// The row/column DIMENSION every rank must agree on -- the one place numPts() is right.
gko::size_type globalRows(const MFFaceCoeffs& matrix)
{
    return static_cast<gko::size_type>(matrix.mesh.ba.numPts());
}

} // namespace

std::shared_ptr<const gko::LinOp> toLinOp(const MFFaceCoeffs& matrix)
{
    // PROTOTYPE (C1): no diagonal is handed over -- the stencil recomputes the centre term
    // inline, so FaceCoeffOp's `diag` parameter is left at its empty default.
    return gko::share(FaceCoeffOp::create(
        makeExecutor(matrix.exec),
        matrix.exec,
        matrix.mesh,
        globalRows(matrix),
        matrix.alpha,
        matrix.upper,
        // The STORED lower, not the interface's: symmetric ALIASES upper here,
        // exactly FaceCoeffOp's convention; `lower` itself is ABSENT.
        matrix.storedLower(),
        // The matrix's `bc`: the ghost reflection it drives is what applies the
        // homogeneous domain BC.
        matrix.bc,
        nullptr
    ));
}

std::shared_ptr<const gko::LinOp>
makeHierarchy(const MFFaceCoeffs& matrix, const SolverConfig& config)
{
    // storedLower() and not `lower` -- the hierarchy wants the ALIASED low side, as toLinOp does.
    return makeFaceCoeffPrecond(
               makeExecutor(matrix.exec),
               globalRows(matrix),
               matrix.alpha,
               matrix.upper,
               matrix.storedLower(),
               matrix.mesh,
               matrix.bc,
               config
    )
        .op;
}

} // namespace blockamr::la
