// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief free standing function implementation of the divergence operator
** ie computes 1/V \sum_f S_f \cdot \phi_f
** where S_f is the face normal flux of a given face
**  phi_f is the face interpolate value
**
**
** @param exec The executor
** @param nInternalFaces - number of internal faces
** @param nBoundaryFaces - number of boundary faces
** @param neighbour - mapping from face id to neighbour cell id
** @param owner - mapping from face id to owner cell id
** @param faceCells - mapping from boundary face id to owner cell id
** @param faceFlux - flux on cell faces
** @param phiF - flux on cell faces
** @param v - cell volumes
** @param res - view holding the result
** @param operatorScaling - any additional coefficients
*/
template<typename ValueType>
void computeDiv(
    const Executor& exec,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    View<const localIdx> neighbour,
    View<const localIdx> owner,
    View<const localIdx> faceCells,
    View<const scalar> faceFlux,
    View<const ValueType> phiF,
    View<const scalar> v,
    View<ValueType> res,
    const dsl::Coeff operatorScaling
)
{
    auto nCells = v.size();

    // Green-Gauss divergence theorem: ∇·(F φ)_C = (1/V_C) * sum_f F_f * φ_f
    //
    // F_f = faceFlux[f] is the signed scalar flux through face f.
    // S_f points from owner to neighbour by construction, so F_f = U · S_f:
    //   F_f > 0 → flux leaving the owner cell and entering the neighbour cell.
    //
    // The DIVERGENCE at a cell measures net outward flux, so:
    //   owner cell:     F_f is outward (S_f points away from owner) → +F_f * φ_f  (add)
    //   neighbour cell: F_f is inward  (S_f points into neighbour)  → −F_f * φ_f  (subtract)
    //
    // This computes +∇·(F φ) (positive divergence form).
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            ValueType flux = faceFlux[i] * phiF[i];
            Kokkos::atomic_add(&res[owner[i]], flux);     // F_f outward from owner
            Kokkos::atomic_sub(&res[neighbour[i]], flux); // F_f inward to neighbour
        },
        "sumFluxesInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + nBoundaryFaces},
        NEON_LAMBDA(const localIdx i) {
            auto own = faceCells[i - nInternalFaces];
            ValueType valueOwn = faceFlux[i] * phiF[i];
            Kokkos::atomic_add(&res[own], valueOwn); // boundary face: F_f outward from owner
        },
        "sumFluxesBoundary"
    );

    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) { res[celli] *= operatorScaling[celli] / v[celli]; },
        "normalizeFluxes"
    );
}

template<typename ValueType>
void computeDivExp(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation<ValueType>& surfInterp,
    Vector<ValueType>& divPhi,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    SurfaceField<ValueType> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh)
    );
    // TODO: remove or implement
    // fill(phif.internalVector(), NeoN::zero<ValueType>::value);
    surfInterp.interpolate(faceFlux, phi, phif);

    // TODO: currently we just copy the boundary values over
    phif.boundaryData().value() = phi.boundaryData().value();

    auto nInternalFaces = mesh.nInternalFaces();
    auto nBoundaryFaces = mesh.nBoundaryFaces();
    computeDiv<ValueType>(
        exec,
        nInternalFaces,
        nBoundaryFaces,
        mesh.faceNeighbour().view(),
        mesh.faceOwner().view(),
        mesh.boundaryMesh().faceCells().view(),
        faceFlux.internalVector().view(),
        phif.internalVector().view(),
        mesh.cellVolumes().view(),
        divPhi.view(),
        operatorScaling

    );
}

#define NF_DECLARE_COMPUTE_EXP_DIV(TYPENAME)                                                       \
    template void computeDivExp<TYPENAME>(                                                         \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        Vector<TYPENAME>&,                                                                         \
        const dsl::Coeff                                                                           \
    )

NF_DECLARE_COMPUTE_EXP_DIV(scalar);
NF_DECLARE_COMPUTE_EXP_DIV(Vec3);

template<typename ValueType>
void computeDivBoundImp(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceField<scalar>& weights,
    const dsl::Coeff operatorScaling
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();

    auto faceFluxV = faceFlux.internalVector().view();

    const auto [ownV, deltaCoeffs] =
        views(mesh.boundaryMesh().faceCells(), mesh.boundaryMesh().deltaCoeffs());

    const auto matIt = ls.faceToMatrixAddress();
    auto const rowOffs = ls.matrix().sparsity()->rowOffs().view();
    auto const diagOffs = matIt->diagOffset().view();

    auto values = ls.matrix().values().view();

    auto [bweights, refGradient, valueFraction, refValue] = views(
        weights.boundaryData().value(),
        phi.boundaryData().refGrad(),
        phi.boundaryData().valueFraction(),
        phi.boundaryData().refValue()
    );

    auto rhs = ls.rhs().view();
    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.boundaryMatrix().values().view();

    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    auto totalFaces = nInternalFaces + nBoundaryFaces;
    parallelFor(
        exec,
        {nInternalFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto bfi = facei - nInternalFaces;
            auto ownRow = ownV[bfi];

            auto ownCoeff = operatorScaling[ownRow];

            auto refValFrac = valueFraction[bfi];
            auto refGradFrac = 1.0 - refValFrac;

            auto flux =
                faceFluxV[facei] * -bweights[bfi] * ownCoeff * refGradFrac * one<ValueType>();

            // Upper triangular - owner offsets
            auto ownRowStart = rowOffs[ownRow];
            auto ownDiagOffs = ownRowStart + static_cast<localIdx>(diagOffs[ownRow]);

            // since upper triangular value is "outside" of system matrix
            // it is stored separately in bMatrix
            bValues[bfi] += flux;
            // diagonal contribution
            Kokkos::atomic_sub(&values[ownDiagOffs], flux);

            // Explicit RHS contribution from the mixed BC:
            //   φ_f = refValFrac * refValue               (Dirichlet part)
            //       + refGradFrac * (φ_C + refGradient/δ)  (Neumann part)
            // The implicit valFrac2 * φ_C term is handled via fluxContrib above.
            // bweights converts the Dirichlet face value to a cell-to-face flux contribution;
            // the Neumann gradient correction (refGradient/δ) enters directly as a known increment.
            auto valueRhs =
                (bweights[bfi] * faceFluxV[facei] * ownCoeff * (refValFrac * refValue[bfi]))
                + refGradFrac * refGradient[bfi] * (1 / deltaCoeffs[bfi]);
            Kokkos::atomic_sub(&rhs[ownRow], valueRhs);
            bRhs[bfi] += valueRhs;
        },
        "computeInterfaceGaussGreenDivCoefficients"
    );
}


template<typename ValueType>
void computeDivIntImp(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceField<scalar>& weights,
    const dsl::Coeff coeff
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const auto [fluxV, weightsV, ownV, neiV, surfFaceCells] = views(
        faceFlux.internalVector(),
        weights.internalVector(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.boundaryMesh().faceCells()
    );
    auto rhs = ls.rhs().view();
    auto values = ls.matrix().values().view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            // row and column indices
            auto ownRow = ownV[facei];
            auto neiRow = neiV[facei];

            // operator sign coefficient  handles: = +/- div
            auto ownCoeff = coeff[ownRow];
            auto neiCoeff = coeff[neiRow];

            // Conservative Gauss-Green divergence assembly.
            // S_f points from owner to neighbour by construction, so F_f < 0 means
            // flux leaves the owner cell and enters the neighbour cell.
            //
            // Decompose face flux via linear interpolation:
            //   ownFluxContrib = w * F_f     — part attributed to the owner cell value
            //   neiFluxContrib = (1-w) * F_f — part attributed to the neighbour cell value
            auto ownFluxContrib = -fluxV[facei] * weightsV[facei] * one<ValueType>();
            auto neiFluxContrib = +fluxV[facei] * (1.0 - weightsV[facei]) * one<ValueType>();

            // triangular coefficients - neighbour -> lower, owner -> upper
            values[ma.lowerIdx(neiRow, facei)] += ownFluxContrib * neiCoeff;
            values[ma.upperIdx(ownRow, facei)] += neiFluxContrib * ownCoeff;

            // diagonal contribution is negative sum of offdiagonal coefficients
            Kokkos::atomic_sub(&values[ma.diagIdx(ownRow)], ownFluxContrib * ownCoeff);
            Kokkos::atomic_sub(&values[ma.diagIdx(neiRow)], neiFluxContrib * neiCoeff);
        },
        "computeLocalGaussGreenDivCoefficients"
    );
};

#define NN_DECLARE_COMPUTE_IMP_DIV(TYPENAME)                                                       \
    template void computeDivIntImp<TYPENAME>(                                                      \
        la::LinearSystem<TYPENAME>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const dsl::Coeff                                                                           \
    );                                                                                             \
    template void computeDivBoundImp<TYPENAME>(                                                    \
        la::LinearSystem<TYPENAME>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const dsl::Coeff                                                                           \
    )

NN_DECLARE_COMPUTE_IMP_DIV(scalar);
NN_DECLARE_COMPUTE_IMP_DIV(Vec3);

};
