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
** @param neighbors - mapping from face id to neighbors cell id
** @param owner - mapping from face id to owner cell id
** @param faceOwners - mapping from boundary face id to owner cell id
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
    View<const localIdx> neighbors,
    View<const localIdx> owners,
    View<const localIdx> faceOwners,
    View<const scalar> faceFlux,
    View<const scalar> bFaceFlux,
    View<const ValueType> phiF,
    View<const ValueType> bPhiF,
    View<const scalar> v,
    View<ValueType> res,
    const dsl::Coeff operatorScaling
)
{
    auto nCells = v.size();

    // Green-Gauss divergence theorem: ∇·(F φ)_C = (1/V_C) * sum_f F_f * φ_f
    //
    // F_f = faceFlux[f] is the signed scalar flux through face f.
    // S_f points from owner to neighbor by construction, so F_f = U · S_f:
    //   F_f > 0 → flux leaving the owner cell and entering the neighbor cell.
    //
    // The DIVERGENCE at a cell measures net outward flux, so:
    //   owner cell:     F_f is outward (S_f points away from owner) → +F_f * φ_f  (add)
    //   neighbor cell: F_f is inward  (S_f points into neighbor)  → −F_f * φ_f  (subtract)
    //
    // This computes +∇·(F φ) (positive divergence form).
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            ValueType flux = faceFlux[i] * phiF[i];
            Kokkos::atomic_add(&res[owners[i]], flux);    // F_f outward from owner
            Kokkos::atomic_sub(&res[neighbors[i]], flux); // F_f inward to neighbor
        },
        "sumFluxesInternal"
    );

    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = faceOwners[bfi];
            ValueType valueOwn = bFaceFlux[bfi] * bPhiF[bfi];
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
        mesh.faceNeighbors().view(),
        mesh.faceOwners().view(),
        mesh.boundaryMesh().faceOwners().view(),
        faceFlux.internalVector().view(),
        faceFlux.boundaryData().value().view(),
        phif.internalVector().view(),
        phif.boundaryData().value().view(),
        mesh.cellVolumes().view(),
        divPhi.view(),
        operatorScaling
    );
}

template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeDivProcBoundImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const SurfaceField<scalar>& weights,
    const dsl::Coeff coeff
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();

    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    if (nProcBoundaryFaces == 0) return;

    const auto [surfFaceCells, isOwner] =
        views(mesh.boundaryMesh().faceOwners(), mesh.boundaryMesh().weights());

    const auto bFluxV = faceFlux.boundaryData().value().view();
    const auto bWeightsV = weights.boundaryData().value().view();
    auto bValues = ls.offDiagonalMatrix().values().view();
    // boundaryMatrix records the diagonal contribution so removeBoundaryContributions can reverse
    // it (proc slots live at [nBoundaryFaces, nBoundaryFaces + nProcBoundaryFaces)).
    auto bndDiagValues = ls.boundaryMatrix().values().view();

    auto values = ls.matrix().values().view();
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    const auto rowOrderV = mesh.boundaryMesh().getRowOrderWriteIndex().view();

    parallelFor(
        exec,
        {0, nProcBoundaryFaces},
        NEON_LAMBDA(const localIdx procFacei) {
            auto bcfacei = nBoundaryFaces + procFacei;
            auto cell = surfFaceCells[bcfacei];

            auto ownCoeff = coeff[cell];

            // S_f points from owner to neighbour by construction; F = faceFlux is signed.
            //
            // From the global computeDivImp for face f (own→nei, weight w = 0 or 1 for upwind):
            //   A[own,own] -= w*F*c         (diagonal of owner)
            //   A[nei,nei] += (1-w)*F*c     (diagonal of neighbour)
            //
            auto isOwnerFace = isOwner[bcfacei] > 0.0;
            auto sign = isOwnerFace ? scalar(-1) : scalar(1);

            auto weight = isOwnerFace ? bWeightsV[bcfacei] : (scalar(1) - bWeightsV[bcfacei]);
            auto fluxContrib = sign * weight * bFluxV[bcfacei] * ownCoeff * one<AssemblyType>();

            Kokkos::atomic_sub(&values[ma.diagIdx(cell)], fluxContrib);
            bndDiagValues[bcfacei] += fluxContrib;
            auto valueOff =
                -sign * (scalar(1) - weight) * bFluxV[bcfacei] * ownCoeff * one<AssemblyType>();
            bValues[rowOrderV[procFacei]] += valueOff;
        },
        "computeProcInterfaceGaussGreenDivCoefficients"
    );
}


template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeDivBoundImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const SurfaceField<scalar>& weights,
    const dsl::Coeff operatorScaling
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const auto [ownV, deltaCoeffs] =
        views(mesh.boundaryMesh().faceOwners(), mesh.boundaryMesh().deltaCoeffs());

    auto values = ls.matrix().values().view();

    auto [bFaceFluxV, bweights, refGradient, valueFraction, refValue] = views(
        faceFlux.boundaryData().value(),
        weights.boundaryData().value(),
        phi.boundaryData().refGrad(),
        phi.boundaryData().valueFraction(),
        phi.boundaryData().refValue()
    );

    auto rhs = ls.rhs().view();
    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.boundaryMatrix().values().view();

    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto ownRow = ownV[bfi];

            auto ownCoeff = operatorScaling[ownRow];

            auto refValFrac = valueFraction[bfi];
            auto refGradFrac = 1.0 - refValFrac;

            auto flux =
                bFaceFluxV[bfi] * -bweights[bfi] * ownCoeff * refGradFrac * one<AssemblyType>();

            // since upper triangular value is "outside" of system matrix
            // it is stored separately in bMatrix
            bValues[bfi] += flux;
            // diagonal contribution
            Kokkos::atomic_sub(&values[ma.diagIdx(ownRow)], flux);

            // Explicit RHS contribution from the mixed BC:
            //   φ_f = refValFrac * refValue               (Dirichlet part)
            //       + refGradFrac * (φ_C + refGradient/δ)  (Neumann part)
            // The implicit valFrac2 * φ_C term is handled via fluxContrib above.
            // bweights converts the Dirichlet face value to a cell-to-face flux contribution;
            // the Neumann gradient correction (refGradient/δ) enters directly as a known increment.
            auto valueRhs =
                (bweights[bfi] * bFaceFluxV[bfi] * ownCoeff * (refValFrac * refValue[bfi]))
                + refGradFrac * refGradient[bfi] * (1 / deltaCoeffs[bfi]);
            Kokkos::atomic_sub(&rhs[ownRow], valueRhs);
            bRhs[bfi] += valueRhs;
        },
        "computeInterfaceGaussGreenDivCoefficients"
    );
}


template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeDivIntImp(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const SurfaceField<scalar>& weights,
    const dsl::Coeff coeff
)
{
    const auto& mesh = phi.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const auto [fluxV, weightsV, ownV, neiV, surfFaceCells] = views(
        faceFlux.internalVector(),
        weights.internalVector(),
        mesh.faceOwners(),
        mesh.faceNeighbors(),
        mesh.boundaryMesh().faceOwners()
    );
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
            // S_f points from owner to neighbor by construction, so F_f < 0 means
            // flux leaves the owner cell and enters the neighbor cell.
            //
            // Decompose face flux via linear interpolation:
            //   ownFluxContrib = w * F_f     — part attributed to the owner cell value
            //   neiFluxContrib = (1-w) * F_f — part attributed to the neighbor cell value
            auto ownFluxContrib = -fluxV[facei] * weightsV[facei] * one<AssemblyType>();
            auto neiFluxContrib = +fluxV[facei] * (1.0 - weightsV[facei]) * one<AssemblyType>();

            // triangular coefficients - neighbor -> lower, owner -> upper
            values[ma.lowerIdx(neiRow, facei)] += ownFluxContrib * neiCoeff;
            values[ma.upperIdx(ownRow, facei)] += neiFluxContrib * ownCoeff;

            // diagonal contribution is negative sum of offdiagonal coefficients
            Kokkos::atomic_sub(&values[ma.diagIdx(ownRow)], ownFluxContrib * ownCoeff);
            Kokkos::atomic_sub(&values[ma.diagIdx(neiRow)], neiFluxContrib * neiCoeff);
        },
        "computeLocalGaussGreenDivCoefficients"
    );
};

template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeDivIntCellBasedImp(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const SurfaceField<scalar>& weights,
    const dsl::Coeff coeff
)
{
    const auto exec = phi.exec();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    auto iterator = std::dynamic_pointer_cast<la::CellBasedIterator>(ls.getMeshIterator()->get());

    const auto [fluxV, weightsV] = views(faceFlux.internalVector(), weights.internalVector());

    auto cellBasedData = iterator->getCellBasedData();
    NF_ASSERT(
        cellBasedData != nullptr,
        "CellBasedData not initialized - call setComputeCellBasedData before invoking the "
        "cell-based kernel"
    );
    auto [cellFacesValues, cellFacesSegments] = cellBasedData->cellFaces.views();
    auto faceSignV = cellBasedData->faceSign.view();
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
                const auto sign = faceSignV[startIdx + i];
                const auto flux = fluxV[faceIdx];
                const auto w = weightsV[faceIdx];

                AssemblyType offDiag;
                AssemblyType diagContrib;
                if (sign > 0) // this cell is the owner: upper-triangle entry
                {
                    offDiag = flux * (1.0 - w) * cellCoeff * one<AssemblyType>();
                    diagContrib = flux * w * cellCoeff * one<AssemblyType>();
                }
                else // this cell is the neighbor: lower-triangle entry
                {
                    offDiag = -flux * w * cellCoeff * one<AssemblyType>();
                    diagContrib = -flux * (1.0 - w) * cellCoeff * one<AssemblyType>();
                }

                values[matrixColumnIdxV[startIdx + i]] += offDiag;
                diagValue += diagContrib;
            }

            values[ma.diagIdx(celli)] += diagValue;
        },
        "fusedKernelCellBased::cellLoop"
    );
}

template<typename FieldValueType, typename AssemblyType>
VolumeField<FieldValueType> GaussGreenDiv<FieldValueType, AssemblyType>::div(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff operatorScaling
) const
{
    std::string name = "div(" + faceFlux.name + "," + phi.name + ")";
    VolumeField<FieldValueType> divPhi(
        this->exec_,
        name,
        this->mesh_,
        createCalculatedBCs<VolumeBoundary<FieldValueType>>(this->mesh_)
    );
    NeoN::fill(divPhi.internalVector(), zero<FieldValueType>());
    NeoN::fill(divPhi.boundaryData().value(), zero<FieldValueType>());
    computeDivExp<FieldValueType>(
        faceFlux, phi, surfaceInterpolation_, divPhi.internalVector(), operatorScaling
    );
    return divPhi;
}

template<typename FieldValueType, typename AssemblyType>
void GaussGreenDiv<FieldValueType, AssemblyType>::div(
    VolumeField<FieldValueType>& divPhi,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff operatorScaling
) const
{
    computeDivExp<FieldValueType>(
        faceFlux, phi, surfaceInterpolation_, divPhi.internalVector(), operatorScaling
    );
}

template<typename FieldValueType, typename AssemblyType>
void GaussGreenDiv<FieldValueType, AssemblyType>::div(
    Vector<FieldValueType>& divPhi,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff operatorScaling
) const
{
    computeDivExp<FieldValueType>(faceFlux, phi, surfaceInterpolation_, divPhi, operatorScaling);
}

/* @brief adds the deferred gradient correction of a corrected scheme (e.g. linearUpwind) to the
** linear-system rhs, i.e. the discrete -operatorScaling * sum_f F_f corr_f per cell. This mirrors
** OpenFOAM's `fvm += fvc::surfaceIntegrate(faceFlux*correction)`: interpolate()=weighted value +
** correction(), the weighted part is assembled implicitly and the correction is an explicit source.
** Boundary correction is zero (physical patches uncorrected), so only internal faces contribute.
*/
template<typename FieldValueType, typename AssemblyType>
void addDivCorrectionToRhs(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const SurfaceField<FieldValueType>& correction,
    const dsl::Coeff operatorScaling
)
{
    const auto& mesh = correction.mesh();
    const auto exec = correction.exec();
    const auto nInternalFaces = mesh.nInternalFaces();

    const auto [fluxV, corrV, ownV, neiV] = views(
        faceFlux.internalVector(),
        correction.internalVector(),
        mesh.faceOwners(),
        mesh.faceNeighbors()
    );
    auto rhs = ls.rhs().view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto own = ownV[facei];
            const auto nei = neiV[facei];
            // sum_f F_f corr_f distributes +F corr to the owner and -F corr to the neighbour
            // (S_f points owner -> neighbour); the term is known so it moves to the rhs negated.
            const FieldValueType contrib = fluxV[facei] * corrV[facei];
            Kokkos::atomic_sub(&rhs[own], operatorScaling[own] * contrib);
            Kokkos::atomic_add(&rhs[nei], operatorScaling[nei] * contrib);
        },
        "addDivCorrectionToRhs"
    );
}

template<typename FieldValueType, typename AssemblyType>
void GaussGreenDiv<FieldValueType, AssemblyType>::div(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff operatorScaling
) const
{
    // TODO boundary weights should be separate so that we can start internal assembly
    const auto weights = surfaceInterpolation_.weight(faceFlux, phi);
    if (auto* cellIter = dynamic_cast<la::CellBasedIterator*>(ls.getMeshIterator()->get().get()))
    {
        if (!cellIter->getCellBasedData())
        {
            cellIter->setComputeCellBasedData(
                phi.mesh(), ls.matrix().sparsity(), ls.faceToMatrixAddress()
            );
        }
        computeDivIntCellBasedImp(ls, faceFlux, phi, weights, operatorScaling);
    }
    else
    {
        computeDivIntImp(ls, faceFlux, phi, weights, operatorScaling);
    }
    computeDivBoundImpl(ls, faceFlux, phi, weights, operatorScaling);
    computeDivProcBoundImpl(ls, faceFlux, phi, weights, operatorScaling);

    // Deferred correction for corrected schemes (e.g. linearUpwind): the implicit matrix uses the
    // upwind weights above, the gradient correction is added explicitly to the rhs.
    if (surfaceInterpolation_.corrected())
    {
        SurfaceField<FieldValueType> correction(
            phi.exec(),
            "divCorrection",
            phi.mesh(),
            createCalculatedBCs<SurfaceBoundary<FieldValueType>>(phi.mesh())
        );
        surfaceInterpolation_.correction(faceFlux, phi, correction);
        addDivCorrectionToRhs(ls, faceFlux, correction, operatorScaling);
    }
}

template class GaussGreenDiv<scalar>;
template class GaussGreenDiv<Vec3>;
template class GaussGreenDiv<Vec3, scalar>;

};
