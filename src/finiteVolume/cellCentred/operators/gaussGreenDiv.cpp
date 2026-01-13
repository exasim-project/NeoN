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
    // check if the executor is GPU
    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (localIdx i = 0; i < nInternalFaces; i++)
        {
            ValueType flux = faceFlux[i] * phiF[i];
            res[owner[i]] += flux;
            res[neighbour[i]] -= flux;
        }

        for (localIdx i = nInternalFaces; i < nInternalFaces + nBoundaryFaces; i++)
        {
            auto own = faceCells[i - nInternalFaces];
            ValueType valueOwn = faceFlux[i] * phiF[i];
            res[own] += valueOwn;
        }

        // TODO does it make sense to store invVol and multiply?
        for (localIdx celli = 0; celli < nCells; celli++)
        {
            res[celli] *= operatorScaling[celli] / v[celli];
        }
    }
    else
    {
        parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx i) {
                ValueType flux = faceFlux[i] * phiF[i];
                Kokkos::atomic_add(&res[owner[i]], flux);
                Kokkos::atomic_sub(&res[neighbour[i]], flux);
            },
            "sumFluxesInternal"
        );

        parallelFor(
            exec,
            {nInternalFaces, nInternalFaces + nBoundaryFaces},
            NEON_LAMBDA(const localIdx i) {
                auto own = faceCells[i - nInternalFaces];
                ValueType valueOwn = faceFlux[i] * phiF[i];
                Kokkos::atomic_add(&res[own], valueOwn);
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
void computeDivImp(
    la::LinearSystem<ValueType, localIdx>& ls,
    const la::MatrixIterator<ValueType>& matIt,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation<ValueType>& surfInterp,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();
    const auto weights = surfInterp.weight(faceFlux, phi);

    const auto
        [faceFluxV,
         weightsV,
         owner,
         neighbour,
         surfFaceCells,
         diagOffs,
         ownOffs,
         neiOffs,
         rowOffs] =
            views(
                faceFlux.internalVector(),
                weights.internalVector(),
                mesh.faceOwner(),
                mesh.faceNeighbour(),
                mesh.boundaryMesh().faceCells(),
                matIt.diagOffset(),
                matIt.ownerOffset(),
                matIt.neighbourOffset(),
                matIt.sparsityPattern()->rowOffs()
            );
    auto [matrix, rhs, bMatrix, bRhs] = ls.view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = owner[facei];
            auto nei = neighbour[facei];

            auto operatorScalingNei = operatorScaling[nei];
            auto operatorScalingOwn = operatorScaling[own];

            auto rowNeiStart = rowOffs[nei];
            auto rowOwnStart = rowOffs[own];

            auto valueUpper = faceFluxV[facei] * -weightsV[facei] * one<ValueType>();
            // matrix.values[matIt.upperIdx(nei, facei)] += valueUpper * operatorScalingNei;
            matrix.values[rowNeiStart + neiOffs[facei]] += valueUpper * operatorScalingNei;
            Kokkos::atomic_sub(
                &matrix.values[rowOwnStart + diagOffs[own]], valueUpper * operatorScalingOwn
            );

            // add owner contribution lower
            auto valueLower = faceFluxV[facei] * (1 - weightsV[facei]) * one<ValueType>();
            // matrix.values[matIt.lowerIdx(own, facei)] += valueLower * operatorScalingOwn;
            //
            matrix.values[rowOwnStart + ownOffs[facei]] += valueLower * operatorScalingOwn;
            Kokkos::atomic_sub(
                &matrix.values[rowNeiStart + diagOffs[nei]], valueLower * operatorScalingNei
            );
        },
        "computeLocalGaussGreenDivCoefficients"
    );

    auto [bweights, refGradient, value, valueFraction, refValue, deltaCoeffs] = views(
        weights.boundaryData().value(),
        phi.boundaryData().refGrad(),
        phi.boundaryData().value(),
        phi.boundaryData().valueFraction(),
        phi.boundaryData().refValue(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    parallelFor(
        exec,
        {nInternalFaces, faceFluxV.size()},
        NEON_LAMBDA(const localIdx facei) {
            auto bcfacei = facei - nInternalFaces;
            auto flux = bweights[bcfacei] * faceFluxV[facei];

            auto own = surfFaceCells[bcfacei];
            auto rowOwnStart = matrix.rowOffs[own];
            auto operatorScalingOwn = operatorScaling[own];

            auto valFrac1 = valueFraction[bcfacei];
            auto valFrac2 = 1.0 - valFrac1;

            auto valueMat = flux * operatorScalingOwn * valFrac2 * one<ValueType>();

            Kokkos::atomic_add(&matrix.values[rowOwnStart + diagOffs[own]], valueMat);
            bMatrix.values[bcfacei] = valueMat;

            auto valueRhs = (flux * operatorScalingOwn * (valFrac1 * refValue[bcfacei]))
                          + valFrac2 * refGradient[bcfacei] * (1 / deltaCoeffs[bcfacei]);

            Kokkos::atomic_sub(&rhs[own], valueRhs);
            bRhs[bcfacei] = valueRhs;
        },
        "computeInterfaceGaussGreenDivCoefficients"
    );
};

#define NN_DECLARE_COMPUTE_IMP_DIV(TYPENAME)                                                       \
    template void computeDivImp<TYPENAME>(                                                         \
        la::LinearSystem<TYPENAME, localIdx>&,                                                     \
        const la::MatrixIterator<TYPENAME>&,                                                       \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const dsl::Coeff                                                                           \
    )

NN_DECLARE_COMPUTE_IMP_DIV(scalar);
NN_DECLARE_COMPUTE_IMP_DIV(Vec3);

};
