// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLaplacianExp(
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const SurfaceField<scalar>&, // gamma,
    const VolumeField<ValueType>& phi,
    Vector<ValueType>& lapPhi,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();

    SurfaceField<ValueType> faceNormalGrad = faceNormalGradient.faceNormalGrad(phi);

    const auto [owners, neighbors, boundaryFaceOwners] =
        views(mesh.faceOwners(), mesh.faceNeighbors(), mesh.boundaryMesh().faceOwners());

    const auto [result, faceAreas, fnGrad, vol] =
        views(lapPhi, mesh.faceAreas(), faceNormalGrad.internalVector(), mesh.cellVolumes());
    const auto fnGradB = faceNormalGrad.boundaryData().value().view();

    auto nInternalFaces = mesh.nInternalFaces();
    auto nBoundaryFaces = mesh.nBoundaryFaces();

    // Green-Gauss Laplacian: ∇·(γ∇φ)_C = (1/V_C) * sum_f γ_f * |S_f| * (∂φ/∂n)_f
    //
    // fnGrad[f] = nonOrthDeltaCoeffs[f] * (phi[nei] − phi[own])  (computed by FaceNormalGradient)
    //   S_f points from owner to neighbour by construction, so fnGrad is the gradient
    //   component in the outward direction from the owner cell.
    //   fnGrad > 0  when phi_N > phi_P (gradient points outward from owner)
    //             → diffusion brings φ into owner → positive Laplacian at owner (owner gains φ)
    //             → diffusion takes φ from neighbour → negative Laplacian at neighbour
    //
    // This computes +∇·(γ∇φ) (positive Laplacian form).
    // TODO use NeoN::add and sub
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            ValueType flux = faceAreas[i] * fnGrad[i];
            Kokkos::atomic_add(
                &result[owners[i]], flux
            ); // +|S_f| * fnGrad (outward gradient from owner)
            Kokkos::atomic_sub(
                &result[neighbors[i]], flux
            ); // −|S_f| * fnGrad (inward gradient for neighbour)
        },
        "computeLaplacianExplicitInternal"
    );

    // Physical (non-proc) boundary faces: only the owner cell is on this rank.
    // For non-proc patches the mesh's full face index and NeoN's compressed
    // boundary index agree (empty patches contribute zero faces),
    // so bFaceAreas[bfi] = mesh.magFaceAreas()[bfi] is correct here.
    const auto bFaceAreas = mesh.boundaryMesh().faceAreas().view();
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = boundaryFaceOwners[bfi];
            ValueType valueOwn = bFaceAreas[bfi] * fnGradB[bfi];
            Kokkos::atomic_add(&result[own], valueOwn);
        },
        "computeLaplacianExplicitBoundary"
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx celli) { result[celli] *= operatorScaling[celli] / vol[celli]; },
        "computeLaplacianExplicitCells"
    );
}

template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeLaplacianProcBoundImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();

    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    if (nProcBoundaryFaces == 0) return;
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const auto [bGammaV, bDeltaCoeffs, boundaryFaceOwner] = views(
        gamma.boundaryData().value(),
        faceNormalGradient.deltaCoeffs().boundaryData().value(),
        mesh.boundaryMesh().faceOwners()
    );
    const auto bcMagSf = mesh.boundaryMesh().faceAreas().view();

    auto bValues = ls.offDiagonalMatrix().values().view();
    // boundaryMatrix records the diagonal contribution so removeBoundaryContributions can reverse
    // it (proc slots live at [nBoundaryFaces, nBoundaryFaces + nProcBoundaryFaces)).
    auto bndDiagValues = ls.boundaryMatrix().values().view();

    auto values = ls.matrix().values().view();
    const auto rowOrderV = mesh.boundaryMesh().getRowOrderWriteIndex().view();

    parallelFor(
        exec,
        {0, nProcBoundaryFaces},
        NEON_LAMBDA(const localIdx procFacei) {
            auto bcfacei = nBoundaryFaces + procFacei;
            auto cell = boundaryFaceOwner[bcfacei];
            auto ownCoeff = coeff[cell];

            auto flux = bGammaV[bcfacei] * bcMagSf[bcfacei] * bDeltaCoeffs[bcfacei];
            auto value = flux * ownCoeff * one<AssemblyType>();

            Kokkos::atomic_sub(&values[ma.diagIdx(cell)], value);
            bValues[rowOrderV[procFacei]] += value;
            bndDiagValues[bcfacei] += value;
        },
        "computeInterfaceLaplacianCoefficients"
    );
}


template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeLaplacianBoundImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff operatorScaling,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();

    const auto [magFaceArea, boundaryFaceOwners] =
        views(mesh.faceAreas(), mesh.boundaryMesh().faceOwners());

    const auto bGammaV = gamma.boundaryData().value().view();
    const auto bDeltaCoeffs = faceNormalGradient.deltaCoeffs().boundaryData().value().view();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    auto values = ls.matrix().values().view();

    auto [refGradient, valueFraction, refValue] = views(
        phi.boundaryData().refGrad(),
        phi.boundaryData().valueFraction(),
        phi.boundaryData().refValue()
    );

    auto rhs = ls.rhs().view();
    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.boundaryMatrix().values().view();

    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto bFaceAreas = mesh.boundaryMesh().faceAreas().view();
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto ownRow = boundaryFaceOwners[bfi];

            auto ownRowCoeff = operatorScaling[ownRow];

            auto refValFrac = valueFraction[bfi];
            auto refGradFrac = 1.0 - refValFrac;
            auto flux = bGammaV[bfi] * bFaceAreas[bfi];
            auto fluxContrib =
                flux * ownRowCoeff * refValFrac * bDeltaCoeffs[bfi] * one<AssemblyType>();

            bValues[bfi] += fluxContrib;
            Kokkos::atomic_sub(&values[ma.diagIdx(ownRow)], fluxContrib);

            auto valueRhs =
                flux * ownRowCoeff
                * (refValFrac * bDeltaCoeffs[bfi] * refValue[bfi] + refGradFrac * refGradient[bfi]);
            Kokkos::atomic_sub(&rhs[ownRow], valueRhs);
            bRhs[bfi] += valueRhs;
        },
        "computeInterfaceLaplacianCoefficients"
    );
}

template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeLaplacianNonOrthCorrImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient
)
{
    if (!faceNormalGradient.hasImplicitCorrection()) return;

    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    const auto nInternalFaces = mesh.nInternalFaces();

    const auto [ownV, neiV] = views(mesh.faceOwners(), mesh.faceNeighbors());
    const auto [gammaV, magFaceArea] = views(gamma.internalVector(), mesh.faceAreas());

    SurfaceField<FieldValueType> corrField(
        exec, "snGradCorr", mesh, createCalculatedBCs<SurfaceBoundary<FieldValueType>>(mesh)
    );
    faceNormalGradient.implicitCorrection(phi, corrField);

    const auto corrV = corrField.internalVector().view();
    auto rhs = ls.rhs().view();

    // Persist the per-internal-face correction flux (OpenFOAM faceFluxCorrectionPtr_ analogue)
    // so the flux reconstruction can add back the SAME deferred correction this assembly used,
    // rather than recomputing it from the post-solve field (which would leave a residual
    // div(phi) = corrDiv(p_before) - corrDiv(p_after) on non-orthogonal meshes). Stored value is
    // the signed face flux out of the owner, corrFlux * coeff[own], matching the matrix-based
    // orthogonal reconstruction's operator scaling.
    //
    // Stored only when the consumer opted in (ls.keepFaceFluxCorrection(), set by the scalar
    // pressure equation that reconstructs the flux) AND the system is scalar. Momentum and
    // turbulence systems never reconstruct flux, so they keep a 0-size placeholder and allocate
    // nothing for the correction.
    const bool storeFfc = ls.keepFaceFluxCorrection() && std::is_same_v<FieldValueType, scalar>;
    auto& ffcPtr = ls.faceFluxCorrection();
    const auto ffcSize = storeFfc ? nInternalFaces : localIdx {0};
    if (!ffcPtr || ffcPtr->size() != ffcSize)
    {
        ffcPtr = std::make_shared<Vector<FieldValueType>>(exec, ffcSize, zero<FieldValueType>());
    }
    auto ffc = ffcPtr->view();

    // Non-orthogonal correction (deferred correction) for corrected / limitedCorrected snGrad.
    // Sign convention: NeoN's Laplacian matrix is the *negative-definite* form (diag<0,
    // off-diag>0; see the atomic_sub on the diagonal in computeLaplacianIntImpl). The deferred
    // correction must enter the RHS with the matching (negative-of-OpenFOAM) sign so the whole
    // assembly stays internally consistent; otherwise the correction is applied with reversed
    // sign relative to the rest of the system, which on snappy/non-orthogonal meshes (motorBike,
    // tiltedCube) corrupts the solved pressure — continuity error climbs each step and the run
    // diverges. With the consistent sign below NeoFOAM converges on par with OpenFOAM.
    //   rhs[own] -= corr[f] * γ_f * |S_f| * coeff[own]
    //   rhs[nei] += corr[f] * γ_f * |S_f| * coeff[nei]
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto corrFlux = corrV[facei] * gammaV[facei] * magFaceArea[facei];
            auto own = ownV[facei];
            auto nei = neiV[facei];
            Kokkos::atomic_sub(&rhs[own], corrFlux * coeff[own]);
            Kokkos::atomic_add(&rhs[nei], corrFlux * coeff[nei]);
            if constexpr (std::is_same_v<FieldValueType, scalar>)
            {
                if (storeFfc)
                {
                    ffc[facei] = corrFlux * coeff[own];
                }
            }
        },
        "computeLaplacianImplicitCorrection"
    );
}

template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeLaplacianIntImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    const auto matIt = ls.faceToMatrixAddress();
    const auto [ownV, neiV, boundaryFaceOwners] =
        views(mesh.faceOwners(), mesh.faceNeighbors(), mesh.boundaryMesh().faceOwners());

    const auto [gammaV, deltaCoeffs, magFaceArea] = views(
        gamma.internalVector(), faceNormalGradient.deltaCoeffs().internalVector(), mesh.faceAreas()
    );

    auto values = ls.matrix().values().view();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const auto nInternalFaces = mesh.nInternalFaces();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            // row and column indices
            auto ownRow = ownV[facei];
            auto neiRow = neiV[facei];

            // operator sign coefficient  handles: = +/- laplacian
            auto ownCoeff = coeff[ownRow];
            auto neiCoeff = coeff[neiRow];

            // Laplacian face coefficient: δ_f · γ_f · |S_f|
            // The Laplacian is symmetric — the same flux value enters both owner and neighbour rows
            // with opposite signs (diffusion out of one cell = diffusion into the other).
            // S_f points from owner to neighbour by construction.
            auto flux =
                deltaCoeffs[facei] * gammaV[facei] * magFaceArea[facei] * one<AssemblyType>();

            // triangular coefficients - neighbour -> lower, owner -> upper
            values[ma.lowerIdx(neiRow, facei)] += flux * neiCoeff;
            values[ma.upperIdx(ownRow, facei)] += flux * ownCoeff;

            // diagonal contribution is negative sum of offdiagonal coefficients
            Kokkos::atomic_sub(&values[ma.diagIdx(ownRow)], flux * ownCoeff);
            Kokkos::atomic_sub(&values[ma.diagIdx(neiRow)], flux * neiCoeff);
        },
        "computeLocalLaplacianCoefficients"
    );
}

template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeLaplacianIntCellBasedImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    auto iterator = std::dynamic_pointer_cast<la::CellBasedIterator>(ls.getMeshIterator()->get());

    const auto [gammaV, deltaCoeffs, magFaceArea] = views(
        gamma.internalVector(), faceNormalGradient.deltaCoeffs().internalVector(), mesh.faceAreas()
    );

    auto cellBasedData = iterator->getCellBasedData();
    NF_ASSERT(
        cellBasedData != nullptr,
        "CellBasedData not initialized - call setComputeCellBasedData before invoking the "
        "cell-based kernel"
    );
    auto [cellFacesValues, cellFacesSegments] = cellBasedData->cellFaces.views();
    auto matrixColumnIdxV = cellBasedData->matrixColumnIdx.view();

    auto values = ls.matrix().values().view();

    parallelFor(
        exec,
        {0, iterator->size()},
        NEON_LAMBDA(const localIdx celli) {
            auto diagValue = zero<AssemblyType>();
            const auto numFaces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
            const auto startIdx = cellFacesSegments[celli];
            const auto cellCoeff = coeff[celli];

            for (localIdx i = 0; i < numFaces; ++i)
            {
                const auto faceIdx = cellFacesValues[startIdx + i];
                // Laplacian is symmetric: flux contribution is identical for owner and neighbor
                const auto offDiag = deltaCoeffs[faceIdx] * gammaV[faceIdx] * magFaceArea[faceIdx]
                                   * cellCoeff * one<AssemblyType>();

                values[matrixColumnIdxV[startIdx + i]] += offDiag;
                diagValue -= offDiag;
            }

            values[ma.diagIdx(celli)] += diagValue;
        },
        "cellBasedLaplacian::cellLoop"
    );
}

template<typename FieldValueType, typename AssemblyType>
void GaussGreenLaplacian<FieldValueType, AssemblyType>::laplacian(
    VolumeField<FieldValueType>& lapPhi,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff
)
{
    computeLaplacianExp<FieldValueType>(
        faceNormalGradient_, gamma, phi, lapPhi.internalVector(), coeff
    );
}

template<typename FieldValueType, typename AssemblyType>
VolumeField<FieldValueType> GaussGreenLaplacian<FieldValueType, AssemblyType>::laplacian(
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff
) const
{
    std::string name = "laplacian(" + gamma.name + "," + phi.name + ")";
    VolumeField<FieldValueType> lapPhi(
        this->exec_,
        name,
        this->mesh_,
        createCalculatedBCs<VolumeBoundary<FieldValueType>>(this->mesh_)
    );
    NeoN::fill(lapPhi.internalVector(), zero<FieldValueType>());
    NeoN::fill(lapPhi.boundaryData().value(), zero<FieldValueType>());
    computeLaplacianExp<FieldValueType>(
        faceNormalGradient_, gamma, phi, lapPhi.internalVector(), coeff
    );
    return lapPhi;
}

template<typename FieldValueType, typename AssemblyType>
void GaussGreenLaplacian<FieldValueType, AssemblyType>::laplacian(
    Vector<FieldValueType>& lapPhi,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff
)
{
    computeLaplacianExp<FieldValueType>(faceNormalGradient_, gamma, phi, lapPhi, coeff);
}

template<typename FieldValueType, typename AssemblyType>
void GaussGreenLaplacian<FieldValueType, AssemblyType>::laplacian(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff
)
{
    if (auto* cellIter = dynamic_cast<la::CellBasedIterator*>(ls.getMeshIterator()->get().get()))
    {
        if (!cellIter->getCellBasedData())
        {
            cellIter->setComputeCellBasedData(
                phi.mesh(), ls.matrix().sparsity(), ls.faceToMatrixAddress()
            );
        }
        computeLaplacianIntCellBasedImpl(ls, gamma, phi, coeff, faceNormalGradient_);
    }
    else
    {
        computeLaplacianIntImpl(ls, gamma, phi, coeff, faceNormalGradient_);
    }
    computeLaplacianBoundImpl(ls, gamma, phi, coeff, faceNormalGradient_);
    computeLaplacianNonOrthCorrImpl(ls, gamma, phi, coeff, faceNormalGradient_);
    computeLaplacianProcBoundImpl(ls, gamma, phi, coeff, faceNormalGradient_);
}


template class GaussGreenLaplacian<scalar>;
template class GaussGreenLaplacian<Vec3>;
template class GaussGreenLaplacian<Vec3, scalar>;

};
