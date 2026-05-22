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

    const auto [result, faceArea, fnGrad, vol] =
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
            ValueType flux = faceArea[i] * fnGrad[i];
            Kokkos::atomic_add(
                &result[owners[i]], flux
            ); // +|S_f| * fnGrad (outward gradient from owner)
            Kokkos::atomic_sub(
                &result[neighbors[i]], flux
            ); // −|S_f| * fnGrad (inward gradient for neighbour)
        },
        "computeLaplacianExplicitInternal"
    );

    // Boundary faces: only the owner cell is on this rank.
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = boundaryFaceOwners[bfi];
            // TODO Issue #515
            ValueType valueOwn = faceArea[nInternalFaces + bfi] * fnGradB[bfi];
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

template<typename ValueType>
void computeLaplacianBoundImpl(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const FaceNormalGradient<ValueType>& faceNormalGradient
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
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto ownRow = boundaryFaceOwners[bfi];

            auto ownRowCoeff = operatorScaling[ownRow];

            auto refValFrac = valueFraction[bfi];
            auto refGradFrac = 1.0 - refValFrac;

            // TODO Issue #515
            auto flux = bGammaV[bfi] * magFaceArea[nInternalFaces + bfi];
            auto fluxContrib =
                flux * ownRowCoeff * refValFrac * bDeltaCoeffs[bfi] * one<ValueType>();

            // since upper triangular value is "outside" of system matrix
            // it is stored separately in bMatrix
            bValues[bfi] += fluxContrib;
            // diagonal contribution
            Kokkos::atomic_sub(&values[ma.diagIdx(ownRow)], fluxContrib);

            // Explicit RHS contribution from the mixed BC:
            //   φ_f = valFrac1 * refValue               (Dirichlet part)
            //       + valFrac2 * (φ_C + refGradient/δ)  (Neumann part)
            // The implicit valFrac2 * φ_C term is handled via fluxContrib above.
            // bweights converts the Dirichlet face value to a cell-to-face flux contribution;
            // the Neumann gradient correction (refGradient/δ) enters directly as a known increment.
            auto valueRhs =
                flux * ownRowCoeff
                * (refValFrac * bDeltaCoeffs[bfi] * refValue[bfi] + refGradFrac * refGradient[bfi]);
            Kokkos::atomic_sub(&rhs[ownRow], valueRhs);
            bRhs[bfi] += valueRhs;
        },
        "computeInterfaceLaplacianCoefficients"
    );
}

template<typename ValueType>
void computeLaplacianNonOrthCorrImpl(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff coeff,
    const FaceNormalGradient<ValueType>& faceNormalGradient
)
{
    if (!faceNormalGradient.hasImplicitCorrection()) return;

    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    const auto nInternalFaces = mesh.nInternalFaces();

    const auto [ownV, neiV] = views(mesh.faceOwners(), mesh.faceNeighbors());
    const auto [gammaV, magFaceArea] = views(gamma.internalVector(), mesh.faceAreas());

    SurfaceField<ValueType> corrField(
        exec, "snGradCorr", mesh, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh)
    );
    faceNormalGradient.implicitCorrection(phi, corrField);

    const auto corrV = corrField.internalVector().view();
    auto rhs = ls.rhs().view();

    // Non-orthogonal correction (deferred correction) for corrected / limitedCorrected snGrad:
    //   rhs[own] += corr[f] * γ_f * |S_f| * coeff[own]
    //   rhs[nei] -= corr[f] * γ_f * |S_f| * coeff[nei]
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto corrFlux = corrV[facei] * gammaV[facei] * magFaceArea[facei];
            auto own = ownV[facei];
            auto nei = neiV[facei];
            Kokkos::atomic_add(&rhs[own], corrFlux * coeff[own]);
            Kokkos::atomic_sub(&rhs[nei], corrFlux * coeff[nei]);
        },
        "computeLaplacianImplicitCorrection"
    );
}

template<typename ValueType>
void computeLaplacianIntImpl(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff coeff,
    const FaceNormalGradient<ValueType>& faceNormalGradient
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
            auto flux = deltaCoeffs[facei] * gammaV[facei] * magFaceArea[facei] * one<ValueType>();

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

template<typename ValueType>
void computeLaplacianIntCellBasedImpl(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff coeff,
    const FaceNormalGradient<ValueType>& faceNormalGradient
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
    auto [cellFacesValues, cellFacesSegments] = cellBasedData->cellFaces.views();
    auto matrixColumnIdxV = cellBasedData->matrixColumnIdx.view();

    auto values = ls.matrix().values().view();

    parallelFor(
        exec,
        {0, iterator->size()},
        NEON_LAMBDA(const localIdx celli) {
            auto diagValue = zero<ValueType>();
            const auto numFaces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
            const auto startIdx = cellFacesSegments[celli];
            const auto cellCoeff = coeff[celli];

            for (localIdx i = 0; i < numFaces; ++i)
            {
                const auto faceIdx = cellFacesValues[startIdx + i];
                // Laplacian is symmetric: flux contribution is identical for owner and neighbor
                const auto offDiag = deltaCoeffs[faceIdx] * gammaV[faceIdx] * magFaceArea[faceIdx]
                                   * cellCoeff * one<ValueType>();

                values[matrixColumnIdxV[startIdx + i]] += offDiag;
                diagValue -= offDiag;
            }

            values[ma.diagIdx(celli)] += diagValue;
        },
        "cellBasedLaplacian::cellLoop"
    );
}

template<typename ValueType>
void GaussGreenLaplacian<ValueType>::laplacian(
    VolumeField<ValueType>& lapPhi,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff coeff
)
{
    computeLaplacianExp<ValueType>(faceNormalGradient_, gamma, phi, lapPhi.internalVector(), coeff);
}

template<typename ValueType>
VolumeField<ValueType> GaussGreenLaplacian<ValueType>::laplacian(
    const SurfaceField<scalar>& gamma, const VolumeField<ValueType>& phi, const dsl::Coeff coeff
) const
{
    std::string name = "laplacian(" + gamma.name + "," + phi.name + ")";
    VolumeField<ValueType> lapPhi(
        this->exec_, name, this->mesh_, createCalculatedBCs<VolumeBoundary<ValueType>>(this->mesh_)
    );
    NeoN::fill(lapPhi.internalVector(), zero<ValueType>());
    NeoN::fill(lapPhi.boundaryData().value(), zero<ValueType>());
    computeLaplacianExp<ValueType>(faceNormalGradient_, gamma, phi, lapPhi.internalVector(), coeff);
    return lapPhi;
}

template<typename ValueType>
void GaussGreenLaplacian<ValueType>::laplacian(
    Vector<ValueType>& lapPhi,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff coeff
)
{
    computeLaplacianExp<ValueType>(faceNormalGradient_, gamma, phi, lapPhi, coeff);
}

template<typename ValueType>
void GaussGreenLaplacian<ValueType>::laplacian(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
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
};

template class GaussGreenLaplacian<scalar>;
template class GaussGreenLaplacian<Vec3>;

};
