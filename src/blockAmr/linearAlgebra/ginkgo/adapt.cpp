// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/linearAlgebra/ginkgo/adapt.hpp"

#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.hpp"
#include "NeoN/blockAmr/linearAlgebra/precond.hpp"

namespace blockamr::la
{

// The row/column DIMENSION every rank must agree on -- the one place numPts() is right.
gko::size_type globalRows(const MFFaceCoeffs& matrix)
{
    return static_cast<gko::size_type>(matrix.mesh.ba.numPts());
}

std::shared_ptr<const gko::LinOp> toLinOp(const MFFaceCoeffs& matrix)
{
    // PROTOTYPE (C1): no diagonal is handed over -- the stencil recomputes the centre term
    // inline, so FaceCoeffOp's `diag` parameter is left at its empty default.
    return gko::share(FaceCoeffOp::create(
        makeExecutor(matrix.exec),
        matrix.exec,
        // storedLower() and not `lower`: symmetric ALIASES upper here, exactly
        // FaceCoeffOp's convention; `lower` itself is ABSENT. The operator's row count
        // comes from mesh.ba -- the globalRows() this file defines.
        FaceCoeffLevel {matrix.alpha, matrix.upper, matrix.storedLower(), matrix.mesh},
        // The matrix's `bc`: the ghost reflection it drives is what applies the
        // homogeneous domain BC. No inhomogeneous datum lives on a matrix.
        DomainBc {matrix.bc}
    ));
}

std::shared_ptr<const gko::LinOp>
makeHierarchy(const MFFaceCoeffs& matrix, const SolverConfig& config)
{
    // storedLower() and not `lower` -- the hierarchy wants the ALIASED low side, as toLinOp does.
    return makeFaceCoeffPrecond(
               makeExecutor(matrix.exec),
               globalRows(matrix),
               FaceCoeffLevel {matrix.alpha, matrix.upper, matrix.storedLower(), matrix.mesh},
               matrix.bc,
               config
    )
        .op;
}

} // namespace blockamr::la
