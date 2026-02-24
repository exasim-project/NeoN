// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDivLaplacian.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeDivLapImplCell(
    la::LinearSystem<ValueType>& ls,
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
    auto matrix = ls.matrix().view();
    const auto sp = ls.faceToMatrixAddress();
    auto cellBasedData = iterator->getCellBasedData();
    auto [cellFacesValues, cellFacesSegments] = cellBasedData->cellFaces.views();
    auto faceNeighbourV = cellBasedData->faceNeighbour.view();
    auto faceSignV = cellBasedData->faceSign.view();
    auto matrixColumnIdxV = cellBasedData->matrixColumnIdx.view();
    const auto [diaOffV, ownOffV, neiOffV] =
        views(sp->diagOffset(), sp->ownerOffset(), sp->neighbourOffset());
    const auto [gammaV, deltaV] =
        views(gamma.internalVector(), faceNormalGradient.deltaCoeffs().internalVector());

    const auto [phiV, /* weightsV,*/ magFaceAreaV] = views(
        phi.internalVector(),
        // weights.internalVector(),
        mesh.magFaceAreas()
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
            matrix.values[diagIdx] += diagValue;
            // rhs[celli] += rhsValue;
        },
        "fusedKernelCellBased::cellLoop"
    );
}


template<typename ValueType>
void computeDivLapImplFace(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    //    const SurfaceInterpolation<ValueType>& lapSurfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();
    //    const auto weights = divSurfInterp.weight(phi, U);
    const auto sp = ls.faceToMatrixAddress();
    // const auto weightsI = lapSurfInterp.weight(phi, U);

    auto matrix = ls.matrix().view();
    auto rhs = ls.rhs().view();
    const auto [gammaV, deltaV] =
        views(gamma.internalVector(), faceNormalGradient.deltaCoeffs().internalVector());
    const auto [diaOffV, ownOffV, neiOffV] =
        views(sp->diagOffset(), sp->ownerOffset(), sp->neighbourOffset());
    const auto [phiV, /* weightsV, */ ownV, neiV, magFaceAreaV] = views(
        phi.internalVector(),
        // weights.internalVector(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.magFaceAreas()
    );

    auto oneV = one<ValueType>();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = ownV[facei];
            auto nei = neiV[facei];

            // auto weight = weightsV[facei];

            auto fluxDiv = phiV[facei];
            const auto weight = (phiV[facei] >= 0) ? 0.0 : 1.0; // weightsV[faceIdx];
            auto fluxLap = deltaV[facei] * gammaV[facei] * magFaceAreaV[facei];

            // add neighbour contribution upper
            auto rowNeiStart = matrix.sparsity.rowOffs[nei];
            auto rowOwnStart = matrix.sparsity.rowOffs[own];

            auto coeffNeiA = 1.0; // coeffA[nei];
            auto coeffOwnA = 1.0; // coeffA[own];
            auto coeffNeiB = 1.0; // coeffB[nei];
            auto coeffOwnB = 1.0; // coeffB[own];

            auto valueDiv = -weight * coeffNeiA * fluxDiv;
            auto valueLap = coeffNeiB * fluxLap;

            auto valueA = valueDiv + valueLap;
            matrix.values[rowNeiStart + neiOffV[facei]] += valueA * oneV;
            Kokkos::atomic_sub(&matrix.values[rowOwnStart + diaOffV[own]], valueA * oneV);

            // upper triangular part
            // add owner contribution lower
            valueDiv = (1 - weight) * coeffOwnA * fluxDiv;
            valueLap = coeffOwnB * fluxLap;
            auto valueB = valueDiv + valueLap;

            matrix.values[rowOwnStart + ownOffV[facei]] += valueB * oneV;
            Kokkos::atomic_sub(&matrix.values[rowNeiStart + diaOffV[nei]], valueB * oneV);
        },
        "computeLocalGaussGreenDivCoefficients"
    );

    const auto surfFaceCells = mesh.boundaryMesh().faceCells().view();
    auto [/*bweights,*/ refGradient, value, valueFraction, refValue, deltaCoeffsA] = views(
        // weights.boundaryData().value(),
        U.boundaryData().refGrad(),
        U.boundaryData().value(),
        U.boundaryData().valueFraction(),
        U.boundaryData().refValue(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.boundaryMatrix().values().view();

    parallelFor(
        exec,
        {nInternalFaces, phiV.size()},
        NEON_LAMBDA(const localIdx facei) {
            auto oneV = one<ValueType>();
            auto bcfacei = facei - nInternalFaces;

            auto fluxDiv = phiV[facei];
            auto fluxLap = gammaV[facei] * magFaceAreaV[facei];

            auto own = surfFaceCells[bcfacei];
            //  auto rowOwnStart = ls.faceToMatrixAddress()->sparsityPattern()->rowOffs[own];
            auto rowOwnStart = matrix.sparsity.rowOffs[own];
            auto coeffAOwn = coeffA[own];
            auto coeffBOwn = coeffB[own];

            auto valFrac1 = valueFraction[bcfacei];
            auto valFrac2 = 1.0 - valFrac1;

            auto bweights = 1.0;

            auto valueDiv = -bweights /*[bcfacei]*/ * coeffAOwn * fluxDiv * valFrac2;
            auto valueLap = deltaV[facei] * coeffBOwn * fluxLap * valFrac1;

            auto valueA = (valueDiv + valueLap) * oneV;

            Kokkos::atomic_sub(&matrix.values[rowOwnStart + diaOffV[own]], valueA);
            bValues[bcfacei] = valueA * (-1.0);

            // div
            auto valueRhsA = ((fluxDiv * coeffAOwn) * (valFrac1 * refValue[bcfacei]))
                           + valFrac2 * refGradient[bcfacei] * (1 / deltaCoeffsA[bcfacei]);
            // lap
            auto valueRhsB =
                fluxLap * coeffBOwn
                * (valFrac1 * refValue[bcfacei] * deltaV[facei] + valFrac2 * refGradient[bcfacei]);

            Kokkos::atomic_sub(&rhs[own], valueRhsA);
            Kokkos::atomic_sub(&rhs[own], valueRhsB);

            bRhs[bcfacei] = (valueRhsA + valueRhsB);
        },
        "computeInterfaceGaussGreenDivCoefficients"
    );
};

#define NN_DECLARE_COMPUTE_IMP_DIV(TYPENAME)                                                       \
    template void computeDivLapImplFace(                                                           \
        la::LinearSystem<TYPENAME>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const dsl::Coeff,                                                                          \
        const dsl::Coeff                                                                           \
    );                                                                                             \
    template void computeDivLapImplCell(                                                           \
        la::LinearSystem<TYPENAME>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const dsl::Coeff,                                                                          \
        const dsl::Coeff,                                                                          \
        std::shared_ptr<la::CellBasedIterator> iterator                                            \
    )

NN_DECLARE_COMPUTE_IMP_DIV(scalar);
NN_DECLARE_COMPUTE_IMP_DIV(Vec3);

};
