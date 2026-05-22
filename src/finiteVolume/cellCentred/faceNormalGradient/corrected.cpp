// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>
#include <type_traits>

#include "NeoN/core/error.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/corrected.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeCorrectedFaceNormalGrad(
    const VolumeField<ValueType>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    SurfaceField<ValueType>& surfaceField
)
{
    if constexpr (std::is_same_v<ValueType, scalar>)
    {
        const UnstructuredMesh& mesh = surfaceField.mesh();
        const auto& exec = surfaceField.exec();

        // Compute cell-centred gradient via Gauss-Green
        GaussGreenGrad gradScheme(exec, mesh);
        VolumeField<Vec3> gradPhi = gradScheme.grad(volField);

        const auto [owners, neighbors, boundaryFaceOwners] =
            views(mesh.faceOwners(), mesh.faceNeighbors(), mesh.boundaryMesh().faceOwners());

        const auto
            [phif,
             phifB,
             phi,
             phiBCValue,
             nonOrthDeltaCoeffs,
             nonOrthDeltaCoeffsB,
             weights,
             corrVec] =
                views(
                    surfaceField.internalVector(),
                    surfaceField.boundaryData().value(),
                    volField.internalVector(),
                    volField.boundaryData().value(),
                    geometryScheme->nonOrthDeltaCoeffs().internalVector(),
                    geometryScheme->nonOrthDeltaCoeffs().boundaryData().value(),
                    geometryScheme->weights().internalVector(),
                    geometryScheme->nonOrthCorrectionVec3s().internalVector()
                );

        const auto gradPhiV = gradPhi.internalVector().view();

        auto nInternalFaces = mesh.nInternalFaces();
        auto nBoundaryFaces = mesh.nBoundaryFaces();

        NeoN::parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx facei) {
                scalar ortho =
                    nonOrthDeltaCoeffs[facei] * (phi[neighbors[facei]] - phi[owners[facei]]);
                Vec3 interpGrad = weights[facei] * gradPhiV[owners[facei]]
                                + (scalar(1) - weights[facei]) * gradPhiV[neighbors[facei]];
                phif[facei] = ortho + (corrVec[facei] & interpGrad);
            },
            "computeCorrectedFaceNormalGradInternal"
        );

        // corrVec is zero at boundaries — boundary snGrad reduces to the uncorrected form.
        NeoN::parallelFor(
            exec,
            {0, nBoundaryFaces},
            NEON_LAMBDA(const localIdx bfi) {
                auto own = boundaryFaceOwners[bfi];
                phifB[bfi] = nonOrthDeltaCoeffsB[bfi] * (phiBCValue[bfi] - phi[own]);
            },
            "computeCorrectedFaceNormalGradBoundary"
        );
    }
    else
    {
        NF_ERROR_EXIT("corrected snGrad for Vec3 fields is not implemented: "
                      "requires tensor gradient support");
    }
}

template<typename ValueType>
void computeCorrectionTerm(
    const VolumeField<ValueType>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    SurfaceField<ValueType>& corrField
)
{
    if constexpr (std::is_same_v<ValueType, scalar>)
    {
        const UnstructuredMesh& mesh = corrField.mesh();
        const auto& exec = corrField.exec();

        GaussGreenGrad gradScheme(exec, mesh);
        VolumeField<Vec3> gradPhi = gradScheme.grad(volField);

        const auto [owners, neighbors] = views(mesh.faceOwners(), mesh.faceNeighbors());

        const auto [corrf, weights, corrVec] = views(
            corrField.internalVector(),
            geometryScheme->weights().internalVector(),
            geometryScheme->nonOrthCorrectionVec3s().internalVector()
        );

        const auto gradPhiV = gradPhi.internalVector().view();
        auto nInternalFaces = mesh.nInternalFaces();

        NeoN::parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx facei) {
                Vec3 interpGrad = weights[facei] * gradPhiV[owners[facei]]
                                + (scalar(1) - weights[facei]) * gradPhiV[neighbors[facei]];
                corrf[facei] = corrVec[facei] & interpGrad;
            },
            "computeCorrectionTermInternal"
        );
        // boundary correction is intentionally not written: the laplacian RHS update
        // iterates only internal faces, so corrField.boundaryData() is never read.
    }
    else
    {
        NF_ERROR_EXIT("implicitCorrection for Vec3 fields is not implemented: "
                      "requires tensor gradient support");
    }
}

#define NF_DECLARE_COMPUTE_CORRECTED_FNG(TYPENAME)                                                 \
    template void computeCorrectedFaceNormalGrad<                                                  \
        TYPENAME>(const VolumeField<TYPENAME>&, const std::shared_ptr<GeometryScheme>, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_CORRECTED_FNG(scalar);
NF_DECLARE_COMPUTE_CORRECTED_FNG(Vec3);

#define NF_DECLARE_COMPUTE_CORRECTION_TERM(TYPENAME)                                               \
    template void computeCorrectionTerm<                                                           \
        TYPENAME>(const VolumeField<TYPENAME>&, const std::shared_ptr<GeometryScheme>, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_CORRECTION_TERM(scalar);
NF_DECLARE_COMPUTE_CORRECTION_TERM(Vec3);

} // namespace NeoN
