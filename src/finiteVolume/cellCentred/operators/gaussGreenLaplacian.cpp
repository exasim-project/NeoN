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

    const auto [owner, neighbour, surfFaceCells] =
        views(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());

    const auto [result, faceArea, fnGrad, vol] =
        views(lapPhi, mesh.magFaceAreas(), faceNormalGrad.internalVector(), mesh.cellVolumes());

    auto nInternalFaces = mesh.nInternalFaces();

    // TODO use NeoN::add and sub
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            ValueType flux = faceArea[i] * fnGrad[i];
            Kokkos::atomic_add(&result[owner[i]], flux);
            Kokkos::atomic_sub(&result[neighbour[i]], flux);
        },
        "computeLaplacianExplicitInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, fnGrad.size()},
        NEON_LAMBDA(const localIdx i) {
            auto own = surfFaceCells[i - nInternalFaces];
            ValueType valueOwn = faceArea[i] * fnGrad[i];
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

#define NF_DECLARE_COMPUTE_EXP_LAP(TYPENAME)                                                       \
    template void computeLaplacianExp<TYPENAME>(                                                   \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        Vector<TYPENAME>&,                                                                         \
        const dsl::Coeff                                                                           \
    )

NF_DECLARE_COMPUTE_EXP_LAP(scalar);
NF_DECLARE_COMPUTE_EXP_LAP(Vec3);

template<typename ValueType>
void computeLaplacianProcBoundImpl(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const FaceNormalGradient<ValueType>& faceNormalGradient
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();

    auto gammaV = gamma.internalVector().view();

    const auto [magFaceArea, surfFaceCells, deltaCoeffs] = views(
        mesh.magFaceAreas(), mesh.boundaryMesh().faceCells(), mesh.boundaryMesh().deltaCoeffs()
    );

    const auto matIt = ls.faceToMatrixAddress();
    auto const rowOffs = matIt->sparsityPattern()->rowOffs().view();
    auto const diagOffs = matIt->diagOffset().view();

    auto values = ls.matrix().values().view();

    auto [/*bweights,*/ refGradient, value, refValue] = views(
        // weights.boundaryData().value(),
        phi.boundaryData().refGrad(),
        phi.boundaryData().value(),
        phi.boundaryData().refValue()
    );

    auto rhs = ls.rhs().view();
    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.boundaryMatrix().values().view();

    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    auto totalFaces = gammaV.size();
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto bcfacei = facei - (nInternalFaces + nBoundaryFaces);
            auto cell = surfFaceCells[bcfacei];
            auto rowStart = rowOffs[cell];
            auto c = operatorScaling[cell];

            auto flux = gammaV[facei] * magFaceArea[facei]; // FIXME  * bweigths[bcfacei];
            auto value = flux * c * one<ValueType>();

            Kokkos::atomic_add(&values[rowStart + diagOffs[cell]], value);
            bValues[bcfacei] += value;
        },
        "computeInterfaceLaplacianCoefficients"
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

    auto gammaV = gamma.internalVector().view();


    const auto [magFaceArea, surfFaceCells, deltaCoeffs] = views(
        mesh.magFaceAreas(),
        mesh.boundaryMesh().faceCells(),
        faceNormalGradient.deltaCoeffs().internalVector()
    );

    const auto matIt = ls.faceToMatrixAddress();
    auto const rowOffs = matIt->sparsityPattern()->rowOffs().view();
    auto const diagOffs = matIt->diagOffset().view();

    auto values = ls.matrix().values().view();

    auto [/*bweights,*/ refGradient, value, valueFraction, refValue] = views(
        // weights.boundaryData().value(),
        phi.boundaryData().refGrad(),
        phi.boundaryData().value(),
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
            auto bcfacei = facei - nInternalFaces;
            auto flux = gammaV[facei] * magFaceArea[facei];

            auto own = surfFaceCells[bcfacei];
            auto rowOwnStart = rowOffs[own];
            auto operatorScalingOwn = operatorScaling[own];

            auto valFrac1 = valueFraction[bcfacei];
            auto valFrac2 = 1.0 - valFrac1;

            // FIXME deltaCoeffs was previously indexed by facei?
            auto valueMat =
                flux * operatorScalingOwn * valFrac2 * deltaCoeffs[facei] * one<ValueType>();

            Kokkos::atomic_sub(&values[rowOwnStart + diagOffs[own]], valueMat);
            bValues[bcfacei] += valueMat;

            ValueType valueRhs = flux * operatorScalingOwn
                               * (valueFraction[bcfacei] * deltaCoeffs[bcfacei] * refValue[bcfacei]
                                  + (1.0 - valueFraction[bcfacei]) * refGradient[bcfacei]);
            Kokkos::atomic_sub(&rhs[own], valueRhs);
            bRhs[bcfacei] = valueRhs;
        },
        "computeInterfaceLaplacianCoefficients"
    );
}

template<typename ValueType>
void computeLaplacianImpl(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const FaceNormalGradient<ValueType>& faceNormalGradient
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    const auto matIt = ls.faceToMatrixAddress();
    const auto [owner, neighbour, surfFaceCells, diagOffs, ownOffs, neiOffs, rowOffs] = views(
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.boundaryMesh().faceCells(),
        matIt->diagOffset(),
        matIt->ownerOffset(),
        matIt->neighbourOffset(),
        matIt->sparsityPattern()->rowOffs()
    );

    const auto [sGamma, deltaCoeffs, magFaceArea] = views(
        gamma.internalVector(),
        faceNormalGradient.deltaCoeffs().internalVector(),
        mesh.magFaceAreas()
    );

    auto rhs = ls.rhs().view();
    auto values = ls.matrix().values().view();

    const auto nInternalFaces = mesh.nInternalFaces();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = owner[facei];
            auto nei = neighbour[facei];

            auto operatorScalingNei = operatorScaling[nei];
            auto operatorScalingOwn = operatorScaling[own];

            // add neighbour contribution upper
            auto rowNeiStart = rowOffs[nei];
            auto rowOwnStart = rowOffs[own];

            auto flux = deltaCoeffs[facei] * sGamma[facei] * magFaceArea[facei];
            // scalar valueNei = (1 - weight) * flux;
            values[rowNeiStart + neiOffs[facei]] += flux * one<ValueType>() * operatorScalingNei;
            Kokkos::atomic_sub(
                &values[rowOwnStart + diagOffs[own]], flux * one<ValueType>() * operatorScalingOwn
            );

            // upper triangular part
            // add owner contribution lower
            values[rowOwnStart + ownOffs[facei]] += flux * one<ValueType>() * operatorScalingOwn;
            Kokkos::atomic_sub(
                &values[rowNeiStart + diagOffs[nei]], flux * one<ValueType>() * operatorScalingNei
            );
        },
        "computeLocalLaplacianCoefficients"
    );
}

#define NN_DECLARE_COMPUTE_IMP_LAP(TYPENAME)                                                                                                                      \
    template void computeLaplacianImpl<                                                                                                                           \
        TYPENAME>(la::LinearSystem<TYPENAME>&, const SurfaceField<scalar>&, const VolumeField<TYPENAME>&, const dsl::Coeff, const FaceNormalGradient<TYPENAME>&); \
    template void computeLaplacianBoundImpl<                                                                                                                      \
        TYPENAME>(la::LinearSystem<TYPENAME>&, const SurfaceField<scalar>&, const VolumeField<TYPENAME>&, const dsl::Coeff, const FaceNormalGradient<TYPENAME>&); \
    template void computeLaplacianProcBoundImpl<                                                                                                                  \
        TYPENAME>(la::LinearSystem<TYPENAME>&, const SurfaceField<scalar>&, const VolumeField<TYPENAME>&, const dsl::Coeff, const FaceNormalGradient<TYPENAME>&)

NN_DECLARE_COMPUTE_IMP_LAP(scalar);
NN_DECLARE_COMPUTE_IMP_LAP(Vec3);

};
