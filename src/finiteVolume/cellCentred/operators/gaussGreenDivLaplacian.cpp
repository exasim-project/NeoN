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
void computeDivLapImpl(
    la::LinearSystem<ValueType, localIdx>& ls,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    const SurfaceInterpolation<ValueType>& lapSurfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB,
    const la::SparsityPattern& sp
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();
    const auto weights = divSurfInterp.weight(phi, U);
    // const auto weightsI = lapSurfInterp.weight(phi, U);

    auto [matrix, rhs] = ls.view();
    const auto [gammaV, deltaV] =
        views(gamma.internalVector(), faceNormalGradient.deltaCoeffs().internalVector());
    const auto [diaOffV, ownOffV, neiOffV] =
        views(sp.diagOffset(), sp.ownerOffset(), sp.neighbourOffset());
    const auto [phiV, weightsV, ownV, neiV, magFaceAreaV] = views(
        phi.internalVector(),
        weights.internalVector(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.magFaceAreas()
    );

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = ownV[facei];
            auto nei = neiV[facei];
            auto oneV = one<ValueType>();

            auto weight = weightsV[facei];
            auto value = zero<ValueType>();

            auto fluxDiv = phiV[facei] * oneV;
            auto fluxLap = deltaV[facei] * gammaV[facei] * magFaceAreaV[facei] * oneV;

            // add neighbour contribution upper
            auto rowNeiStart = matrix.rowOffs[nei];
            auto rowOwnStart = matrix.rowOffs[own];

            auto coeffNeiA = coeffA[nei];
            auto coeffOwnA = coeffA[own];
            auto coeffNeiB = coeffB[nei];
            auto coeffOwnB = coeffB[own];

            auto valueDiv = -weight * coeffNeiA * fluxDiv;
            auto valueLap = coeffNeiB * fluxLap;

            auto valueA = valueDiv + valueLap;
            matrix.values[rowNeiStart + neiOffV[facei]] += valueA;
            Kokkos::atomic_sub(&matrix.values[rowOwnStart + diaOffV[own]], valueA);

            // upper triangular part
            // add owner contribution lower
            valueDiv = (1 - weight) * coeffOwnA * fluxDiv;
            valueLap = coeffOwnB * fluxLap;
            auto valueB = valueDiv + valueLap;

            matrix.values[rowOwnStart + ownOffV[facei]] += valueB;
            Kokkos::atomic_sub(&matrix.values[rowNeiStart + diaOffV[nei]], valueB);
        },
        "computeLocalGaussGreenDivCoefficients"
    );

    const auto surfFaceCells = mesh.boundaryMesh().faceCells().view();
    // auto [bweights, refGradient, value, valueFraction, refValue, deltaCoeffs] = views(
    //     weights.boundaryData().value(),
    //     phi.boundaryData().refGrad(),
    //     phi.boundaryData().value(),
    //     phi.boundaryData().valueFraction(),
    //     phi.boundaryData().refValue(),
    //     mesh.boundaryMesh().deltaCoeffs()
    // );

    // auto& bcCoeffs =
    //     ls.auxiliaryCoefficients().template get<la::BoundaryCoefficients<ValueType, localIdx>>(
    //         "boundaryCoefficients"
    //     );

    // auto [boundValues, rhsBoundValues] = views(bcCoeffs.matrixValues, bcCoeffs.rhsValues);

    // parallelFor(
    //     exec,
    //     {nInternalFaces, faceFluxV.size()},
    //     NEON_LAMBDA(const localIdx facei) {
    //         auto bcfacei = facei - nInternalFaces;
    //         auto flux = bweights[bcfacei] * faceFluxV[facei];

    //         auto own = surfFaceCells[bcfacei];
    //         auto rowOwnStart = matrix.rowOffs[own];
    //         auto operatorScalingOwn = operatorScaling[own];

    //         auto valFrac1 = valueFraction[bcfacei];
    //         auto valFrac2 = 1.0 - valFrac1;

    //         auto valueMat = flux * operatorScalingOwn * valFrac2 * one<ValueType>();

    //         Kokkos::atomic_add(&matrix.values[rowOwnStart + diagOffs[own]], valueMat);
    //         boundValues[bcfacei] = valueMat;

    //         auto valueRhs = (flux * operatorScalingOwn * (valFrac1 * refValue[bcfacei]))
    //                       + valFrac2 * refGradient[bcfacei] * (1 / deltaCoeffs[bcfacei]);

    //         Kokkos::atomic_sub(&rhs[own], valueRhs);

    //         rhsBoundValues[bcfacei] = valueRhs;
    //     },
    //     "computeInterfaceGaussGreenDivCoefficients"
    // );
};

#define NN_DECLARE_COMPUTE_IMP_DIV(TYPENAME)                                                       \
    template void computeDivLapImpl(la::LinearSystem<TYPENAME, localIdx>&, const VolumeField<TYPENAME>&, const SurfaceField<scalar>&, const SurfaceField<scalar>&, const SurfaceInterpolation<TYPENAME>&, const SurfaceInterpolation<TYPENAME>&, const FaceNormalGradient<TYPENAME>&, const dsl::Coeff, const dsl::Coeff, const la::SparsityPattern&)

NN_DECLARE_COMPUTE_IMP_DIV(scalar);
NN_DECLARE_COMPUTE_IMP_DIV(Vec3);

};
