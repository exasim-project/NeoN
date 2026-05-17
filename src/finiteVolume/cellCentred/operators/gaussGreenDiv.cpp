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
void computeDivProcBoundImpl(
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

    const auto matIt = ls.faceToMatrixAddress();
    const auto [rowOffs, diagOffs] =
        views(matIt->sparsityPattern()->rowOffs(), matIt->diagOffset());

    const auto [surfFaceCells, isOwner] =
        views(mesh.boundaryMesh().faceCells(), mesh.boundaryMesh().weights());

    const auto [bweights] = views(weights.internalVector());

    auto bValues = ls.nonLocalMatrix().values().view();
    auto values = ls.matrix().values().view();

    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    auto totalFaces = faceFluxV.size();
    NeoN::mpi::Environment mpiEnviron;
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto bcfacei = facei - (nInternalFaces);
            // FIXME this is weird needing two indices
            auto bcfaceii = facei - (nInternalFaces + nBoundaryFaces);
            auto cell = surfFaceCells[bcfacei];
            auto rowStart = rowOffs[cell];
            auto c = operatorScaling[cell];

            // Conservative upwind divergence for processor boundary faces.
            // S_f points from owner to neighbour by construction; F = faceFlux is signed.
            //
            // From the global computeDivImp for face f (own→nei, weight w = 0 or 1 for upwind):
            //   A[own,own] -= w*F*c         (diagonal of owner)
            //   A[own,nei] -= (1-w)*F*c     (off-diagonal: owner row, nei column)
            //   A[nei,own] += w*F*c         (off-diagonal: nei row, own column)
            //   A[nei,nei] += (1-w)*F*c     (diagonal of neighbour)
            //
            // Each rank uses the raw upwind weight: w_raw = (F >= 0) ? 1 : 0.
            // The owner's diagonal coefficient is w_raw; the non-owner's is (1 - w_raw).
            auto isOwnerFace = isOwner[bcfacei] > 0.0;
            auto sign = isOwnerFace ? scalar(-1) : scalar(1);
            auto w_raw = bweights[facei]; // use global face index, not boundary-local bcfacei
            // Diagonal weight: owner uses w_raw, non-owner uses (1-w_raw)
            auto w_diag = isOwnerFace ? w_raw : (scalar(1) - w_raw);
            auto F = faceFluxV[facei];
            auto value = sign * w_diag * F * c * one<ValueType>();

            Kokkos::atomic_sub(&values[rowStart + diagOffs[cell]], value);
            // bValues[bcfaceii] += value ; // this will be

            // Off-diagonal (ghost coupling).
            //
            // Div is asymmetric: in the global computeDivImp the two off-diagonals
            // around face (own, nei) are
            //     M[own, nei] = +F * (1 - w) * c       (upper)
            //     M[nei, own] = -F *  w      * c       (lower)
            // i.e. they have opposite signs.
            //
            // On the owner-side rank we are storing M[local=own, ghost=nei]:
            //   sign = -1, w_diag = w_raw, so we want valueOff = +F*(1-w_raw)*c
            // On the non-owner-side rank we are storing M[local=nei, ghost=own]:
            //   sign = +1, w_diag = (1 - w_raw), so we want valueOff = -F*w_raw*c
            //                                              = -F*(1 - w_diag)*c
            //
            // Both cases are captured by negating `sign` in front of the magnitude:
            //   owner:     -(-1) * (1 - w_raw)        = +(1 - w)
            //   non-owner: -(+1) * (1 - (1-w_raw))    = -w
            // This mirrors the diag formula `value = sign * w_diag * F * c`, which
            // already encodes the asymmetric +diag[own]/-diag[nei] split correctly.
            auto valueOff = -sign * (scalar(1) - w_diag) * F * c * one<ValueType>();
            bValues[bcfaceii] += valueOff;
        },
        "computeProcInterfaceGaussGreenDivCoefficients"
    );
}


template<typename ValueType>
void computeDivBoundImpl(
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
    auto const rowOffs = matIt->sparsityPattern()->rowOffs().view();
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
    const auto& matIt = ls.faceToMatrixAddress();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();

    const auto [fluxV, weightsV, ownV, neiV, surfFaceCells, diagOffs, ownOffs, neiOffs, rowOffs] =
        views(
            faceFlux.internalVector(),
            weights.internalVector(),
            mesh.faceOwner(),
            mesh.faceNeighbour(),
            mesh.boundaryMesh().faceCells(),
            matIt->diagOffset(),
            matIt->ownerOffset(),
            matIt->neighbourOffset(),
            matIt->sparsityPattern()->rowOffs()
        );
    auto rhs = ls.rhs().view();
    auto values = ls.matrix().values().view();

    NeoN::mpi::Environment mpiEnviron;
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            // row and column indices
            auto ownRow = ownV[facei];
            auto neiRow = neiV[facei];

            auto ownRowStart = rowOffs[ownRow];
            auto neiRowStart = rowOffs[neiRow];

            // operator sign coefficient  handles: = +/- div
            auto ownCoeff = coeff[ownRow];
            auto neiCoeff = coeff[neiRow];

            // matrix value diagonal and column offsets
            // NOTE TODO these are currently hardcode COO/CSR offsets
            auto ownDiagOffs = ownRowStart + static_cast<localIdx>(diagOffs[ownRow]);
            auto neiDiagOffs = neiRowStart + static_cast<localIdx>(diagOffs[neiRow]);
            auto upperColOffs = ownRowStart + ownOffs[facei];
            auto lowerColOffs = neiRowStart + neiOffs[facei];

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
            values[lowerColOffs] += ownFluxContrib * neiCoeff;
            values[upperColOffs] += neiFluxContrib * ownCoeff;

            // diagonal contribution is negative sum of offdiagonal coefficients
            Kokkos::atomic_sub(&values[ownDiagOffs], ownFluxContrib * ownCoeff);
            Kokkos::atomic_sub(&values[neiDiagOffs], neiFluxContrib * neiCoeff);
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
    template void computeDivBoundImpl<TYPENAME>(                                                   \
        la::LinearSystem<TYPENAME>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const dsl::Coeff                                                                           \
    );                                                                                             \
    template void computeDivProcBoundImpl<TYPENAME>(                                               \
        la::LinearSystem<TYPENAME>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const dsl::Coeff                                                                           \
    )

NN_DECLARE_COMPUTE_IMP_DIV(scalar);
NN_DECLARE_COMPUTE_IMP_DIV(Vec3);

template class GaussGreenDiv<scalar>;
template class GaussGreenDiv<Vec3>;

};
