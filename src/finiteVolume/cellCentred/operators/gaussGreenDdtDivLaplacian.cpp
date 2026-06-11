// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDivLaplacian.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeDdtDivLapImplCell(
    la::LinearSystem<ValueType>& ls,
    scalar dt,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    //    const SurfaceInterpolation<ValueType>& lapSurfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB,
    std::shared_ptr<la::CellBasedIterator> iterator
)
{
    auto exec = ls.exec();
    const auto& mesh = phi.mesh();
    const auto vol = mesh.cellVolumes().view();
    auto matrix = ls.matrix().view();
    const auto sp = ls.faceToMatrixAddress();
    auto cellBasedData = iterator->getCellBasedData();
    NF_ASSERT(
        cellBasedData != nullptr,
        "CellBasedData not initialized - call setComputeCellBasedData before invoking the "
        "cell-based kernel"
    );
    auto [cellFacesValues, cellFacesSegments] = cellBasedData->cellFaces.views();
    auto faceNeighbourV = cellBasedData->faceNeighbour.view();
    auto faceSignV = cellBasedData->faceSign.view();
    auto matrixColumnIdxV = cellBasedData->matrixColumnIdx.view();

    // const auto operatorScaling = this->getCoefficient();
    const auto diagOffs = ls.faceToMatrixAddress()->diagOffset().view();
    const auto oldVector = oldTime(U).internalVector().view();
    auto [rhs, values] = views(ls.rhs(), ls.matrix().values());
    auto [colIdx, rowOffs] = ls.matrix().sparsity()->view();

    const scalar a0a1 = 1.0 / dt;

    const auto [diaOffV, ownOffV, neiOffV] =
        views(sp->diagOffset(), sp->ownerOffset(), sp->neighbourOffset());
    const auto [gammaV, deltaV] =
        views(gamma.internalVector(), faceNormalGradient.deltaCoeffs().internalVector());

    const auto [phiV, /* weightsV,*/ magFaceAreaV] = views(
        phi.internalVector(),
        // weights.internalVector(),
        mesh.faceAreas()
    );

    parallelFor(
        exec,
        {0, iterator->size()},
        NEON_LAMBDA(const localIdx celli) {
            // DDT contribution to diagonal
            const auto diagIdx = matrix.sparsity.rowOffs[celli] + diaOffV[celli];
            // const auto coeff = a0 * vol[celli];
            // auto diagValue = coeff * one<ValueType>();
            // auto rhsValue = coeff * oldVector[celli];

            // Loop over faces of this cell
            auto diagValue = zero<ValueType>();
            const auto numFaces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
            const auto startIdx = cellFacesSegments[celli];

            for (localIdx i = 0; i < numFaces; ++i)
            {
                const auto faceIdx = cellFacesValues[startIdx + i];
                const auto neiCell = faceNeighbourV[startIdx + i];
                const auto sign = faceSignV[startIdx + i];

                // Compute flux on-the-fly
                const auto fluxDiv = phiV[faceIdx]; // faceFluxV[faceIdx];
                // FIXME;
                const auto weight = (phiV[faceIdx] >= 0) ? 0.0 : 1.0; // weightsV[faceIdx];
                const auto lapFlux = deltaV[faceIdx] * gammaV[faceIdx] * magFaceAreaV[faceIdx];
                const auto combinedFlux = (-weight * fluxDiv + lapFlux) * one<ValueType>();

                const auto offDiagValue = sign * combinedFlux;
                matrix.values[matrixColumnIdxV[startIdx + i]] += offDiagValue;

                // Contribution to diagonal (subtract off-diagonal)
                diagValue -= offDiagValue;
            }

            // Write diagonal and RHS
            matrix.values[diagIdx] += diagValue + a0a1 * one<ValueType>();
            // FIXME
            // const auto commonCoef = operatorScaling[celli] * vol[celli];
            rhs[celli] += a0a1 * oldVector[celli];
        },
        "fusedKernelCellBased::cellLoop"
    );
}


#define NN_DECLARE_COMPUTE_IMP_DDTDIVLAP(TYPENAME)                                                 \
    template void computeDdtDivLapImplCell(                                                        \
        la::LinearSystem<TYPENAME>&,                                                               \
        scalar,                                                                                    \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const dsl::Coeff,                                                                          \
        const dsl::Coeff,                                                                          \
        std::shared_ptr<la::CellBasedIterator> iterator                                            \
    )

NN_DECLARE_COMPUTE_IMP_DDTDIVLAP(scalar);
NN_DECLARE_COMPUTE_IMP_DDTDIVLAP(Vec3);

};
