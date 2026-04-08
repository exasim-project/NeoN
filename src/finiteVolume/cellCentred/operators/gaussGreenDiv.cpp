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

            // NOTE here only bweights is taken
            // thus the other side has to make sure that
            // bweights_this = (1 - bweights_other)
            auto alpha = isOwner[bcfacei] > 0.0 ? -bweights[bcfacei] : 1.0 - bweights[bcfacei];
            auto flux = alpha * faceFluxV[facei];
            auto value = flux * c * one<ValueType>();

            Kokkos::atomic_sub(&values[rowStart + diagOffs[cell]], value);
            bValues[bcfaceii] += value;
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

    const auto [surfFaceCells, deltaCoeffs] =
        views(mesh.boundaryMesh().faceCells(), mesh.boundaryMesh().deltaCoeffs());

    const auto matIt = ls.faceToMatrixAddress();
    auto const rowOffs = matIt->sparsityPattern()->rowOffs().view();
    auto const diagOffs = matIt->diagOffset().view();

    auto values = ls.matrix().values().view();

    auto [bweights, refGradient, value, valueFraction, refValue] = views(
        weights.boundaryData().value(),
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
            auto flux = bweights[bcfacei] * faceFluxV[facei];

            auto own = surfFaceCells[bcfacei];
            auto rowOwnStart = rowOffs[own];
            auto operatorScalingOwn = operatorScaling[own];

            auto valFrac1 = valueFraction[bcfacei];
            auto valFrac2 = 1.0 - valFrac1;

            auto valueMat = -flux * operatorScalingOwn * valFrac2 * one<ValueType>();
            Kokkos::atomic_add(&values[rowOwnStart + diagOffs[own]], valueMat);
            bValues[bcfacei] += valueMat;

            auto valueRhs = (flux * operatorScalingOwn * (valFrac1 * refValue[bcfacei]))
                          + valFrac2 * refGradient[bcfacei] * (1 / deltaCoeffs[bcfacei]);

            Kokkos::atomic_sub(&rhs[own], valueRhs);
            bRhs[bcfacei] = valueRhs;
        },
        "computeInterfaceGaussGreenDivCoefficients"
    );
}


template<typename ValueType>
void computeDivImp(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceField<scalar>& weights,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto& matIt = ls.faceToMatrixAddress();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();

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
            auto own = owner[facei];
            auto nei = neighbour[facei];

            auto operatorScalingNei = operatorScaling[nei];
            auto operatorScalingOwn = operatorScaling[own];

            auto rowNeiStart = rowOffs[nei];
            auto rowOwnStart = rowOffs[own];

            // Conservative Gauss-Green divergence assembly.
            // S_f points from owner to neighbour by construction, so F_f > 0 means
            // flux leaves the owner cell and enters the neighbour cell.
            //
            // Decompose face flux via linear interpolation:
            //   ownFluxContrib = w * F_f       — part attributed to the owner cell value
            //   neiFluxContrib = (1-w) * F_f   — part attributed to the neighbour cell value

            auto ownFluxContrib = faceFluxV[facei] * weightsV[facei] * one<ValueType>();
            // A[nei, own] — lower triangular: ownFluxContrib enters neighbour → positive sign
            values[rowNeiStart + neiOffs[facei]] += ownFluxContrib * operatorScalingNei;
            // values[matIt->lowerIdx(nei, facei)] += ownFluxContrib * operatorScalingNei;
            // diag[own] — ownFluxContrib leaves owner → negative sign
            Kokkos::atomic_sub(
                &values[rowOwnStart + static_cast<int>(diagOffs[own])],
                ownFluxContrib * operatorScalingOwn
            );

            auto neiFluxContrib = faceFluxV[facei] * (1.0 - weightsV[facei]) * one<ValueType>();
            // A[own, nei] — upper triangular: neiFluxContrib leaves owner → negative sign
            values[rowOwnStart + ownOffs[facei]] -= neiFluxContrib * operatorScalingOwn;
            // values[matIt->upperIdx(own, facei)] -= neiFluxContrib * operatorScalingOwn;
            // diag[nei] — neiFluxContrib enters neighbour → positive sign
            values[rowNeiStart + diagOffs[nei]] += neiFluxContrib * operatorScalingNei;
        },
        "computeLocalGaussGreenDivCoefficients"
    );
};

#define NN_DECLARE_COMPUTE_IMP_DIV(TYPENAME)                                                       \
    template void computeDivImp<TYPENAME>(                                                         \
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

};
