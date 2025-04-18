// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors


#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLaplacianExp(
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const SurfaceField<scalar>&, // gamma,
    VolumeField<ValueType>& phi,
    Vector<ValueType>& lapPhi,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();

    SurfaceField<ValueType> faceNormalGrad = faceNormalGradient.faceNormalGrad(phi);

    const auto [owner, neighbour, surfFaceCells] =
        spans(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());

    const auto [result, faceArea, fnGrad, vol] =
        spans(lapPhi, mesh.magFaceAreas(), faceNormalGrad.internalVector(), mesh.cellVolumes());

    auto nInternalFaces = mesh.nInternalFaces();

    // TODO use NeoN::add and sub
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx i) {
            ValueType flux = faceArea[i] * fnGrad[i];
            Kokkos::atomic_add(&result[owner[i]], flux);
            Kokkos::atomic_sub(&result[neighbour[i]], flux);
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, fnGrad.size()},
        KOKKOS_LAMBDA(const localIdx i) {
            auto own = surfFaceCells[i - nInternalFaces];
            ValueType valueOwn = faceArea[i] * fnGrad[i];
            Kokkos::atomic_add(&result[own], valueOwn);
        }
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        KOKKOS_LAMBDA(const localIdx celli) {
            result[celli] *= operatorScaling[celli] / vol[celli];
        }
    );
}

#define NF_DECLARE_COMPUTE_EXP_LAP(TYPENAME)                                                       \
    template void computeLaplacianExp<TYPENAME>(                                                   \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const SurfaceField<scalar>&,                                                               \
        VolumeField<TYPENAME>&,                                                                    \
        Vector<TYPENAME>&,                                                                         \
        const dsl::Coeff                                                                           \
    )

NF_DECLARE_COMPUTE_EXP_LAP(scalar);
NF_DECLARE_COMPUTE_EXP_LAP(Vec3);


template<typename ValueType>
void computeLaplacianImpl(
    la::LinearSystem<ValueType, localIdx>& ls,
    const SurfaceField<scalar>& gamma,
    VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const SparsityPattern& sparsityPattern,
    const FaceNormalGradient<ValueType>& faceNormalGradient
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();
    const auto [owner, neighbour, surfFaceCells, diagOffs, ownOffs, neiOffs] = spans(
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.boundaryMesh().faceCells(),
        sparsityPattern.diagOffset(),
        sparsityPattern.ownerOffset(),
        sparsityPattern.neighbourOffset()
    );

    const auto [sGamma, deltaCoeffs, magFaceArea] = spans(
        gamma.internalVector(),
        faceNormalGradient.deltaCoeffs().internalVector(),
        mesh.magFaceAreas()
    );

    // FIXME: what if order changes
    auto [values, colIdxs, rowPtrs] = ls.matrix().view();

    // const auto rowPtrs = ls.matrix().rowPtrs();
    // const auto colIdxs = ls.matrix().colIdxs();
    // auto values = ls.matrix().values().span();
    auto rhs = ls.rhs().view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            scalar flux = deltaCoeffs[facei] * sGamma[facei] * magFaceArea[facei];

            auto own = owner[facei];
            auto nei = neighbour[facei];

            // add neighbour contribution upper
            auto rowNeiStart = rowPtrs[nei];
            auto rowOwnStart = rowPtrs[own];

            scalar operatorScalingNei = operatorScaling[nei];
            scalar operatorScalingOwn = operatorScaling[own];

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
        }
    );

    auto [refGradient, value, valueFraction, refValue] = spans(
        phi.boundaryData().refGrad(),
        phi.boundaryData().value(),
        phi.boundaryData().valueFraction(),
        phi.boundaryData().refValue()
    );

    parallelFor(
        exec,
        {nInternalFaces, sGamma.size()},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto bcfacei = facei - nInternalFaces;
            auto flux = sGamma[facei] * magFaceArea[facei];

            auto own = surfFaceCells[bcfacei];
            auto rowOwnStart = rowPtrs[own];
            auto operatorScalingOwn = operatorScaling[own];

            values[rowOwnStart + diagOffs[own]] -= flux * operatorScalingOwn
                                                 * valueFraction[bcfacei] * deltaCoeffs[facei]
                                                 * one<ValueType>();
            rhs[own] -=
                (flux * operatorScalingOwn
                 * (valueFraction[bcfacei] * deltaCoeffs[facei] * refValue[bcfacei]
                    + (1.0 - valueFraction[bcfacei]) * refGradient[bcfacei]));
        }
    );
}

#define NF_DECLARE_COMPUTE_IMP_LAP(TYPENAME)                                                       \
    template void computeLaplacianImpl<                                                            \
        TYPENAME>(la::LinearSystem<TYPENAME, localIdx>&, const SurfaceField<scalar>&, VolumeField<TYPENAME>&, const dsl::Coeff, const SparsityPattern&, const FaceNormalGradient<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_LAP(scalar);
NF_DECLARE_COMPUTE_IMP_LAP(Vec3);

};
