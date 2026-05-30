// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>
#include <type_traits>

#include "NeoN/core/error.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/corrected.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/processorGradientHalo.hpp"

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
                phif[facei] = ortho + (interpGrad & corrVec[facei]);
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

#ifdef NF_WITH_MPI_SUPPORT
        // Processor faces: full non-orthogonal correction, matching OpenFOAM (v2b / N4):
        //   snGrad = nonOrthDeltaCoeffs*(phiNei - phiOwn) + corrVec . interpolate(grad).
        // The interpolated cell gradient needs the neighbour cell gradient across the rank
        // boundary, which is halo-exchanged here.
        auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
        if (nProcBoundaryFaces > 0)
        {
            const auto gradNei =
                detail::exchangeProcNeighbourGradient(exec, mesh, gradPhi.internalVector());
            const auto [weightsB, corrVecB, gradNeiV] = views(
                geometryScheme->weights().boundaryData().value(),
                geometryScheme->nonOrthCorrectionVec3s().boundaryData().value(),
                gradNei
            );
            NeoN::parallelFor(
                exec,
                {0, nProcBoundaryFaces},
                NEON_LAMBDA(const localIdx procFacei) {
                    auto bcfacei = nBoundaryFaces + procFacei;
                    auto own = boundaryFaceOwners[bcfacei];
                    scalar ortho = nonOrthDeltaCoeffsB[bcfacei] * (phiBCValue[bcfacei] - phi[own]);
                    Vec3 interpGrad = weightsB[bcfacei] * gradPhiV[own]
                                    + (scalar(1) - weightsB[bcfacei]) * gradNeiV[procFacei];
                    phifB[bcfacei] = ortho + (interpGrad & corrVecB[bcfacei]);
                },
                "computeCorrectedFaceNormalGradProcBoundary"
            );
        }
#endif
    }
    else if constexpr (std::is_same_v<ValueType, Vec3>)
    {
        const UnstructuredMesh& mesh = surfaceField.mesh();
        const auto& exec = surfaceField.exec();

        GaussGreenGrad gradScheme(exec, mesh);
        VolumeField<Tensor> gradPhi = gradScheme.gradTensor(volField);

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
                Vec3 ortho =
                    nonOrthDeltaCoeffs[facei] * (phi[neighbors[facei]] - phi[owners[facei]]);
                Tensor interpGrad = weights[facei] * gradPhiV[owners[facei]]
                                  + (scalar(1) - weights[facei]) * gradPhiV[neighbors[facei]];
                phif[facei] = ortho + (interpGrad & corrVec[facei]);
            },
            "computeCorrectedFaceNormalGradInternalVec3"
        );

        NeoN::parallelFor(
            exec,
            {0, nBoundaryFaces},
            NEON_LAMBDA(const localIdx bfi) {
                auto own = boundaryFaceOwners[bfi];
                phifB[bfi] = nonOrthDeltaCoeffsB[bfi] * (phiBCValue[bfi] - phi[own]);
            },
            "computeCorrectedFaceNormalGradBoundaryVec3"
        );

#ifdef NF_WITH_MPI_SUPPORT
        // Processor faces: full non-orthogonal correction (v2b / N4), matching OpenFOAM's
        // component-wise corrected snGrad. interpGrad is the interpolated cell gradient tensor;
        // the neighbour gradient is halo-exchanged. (interpGrad & corrVec) is a Vec3.
        auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
        if (nProcBoundaryFaces > 0)
        {
            const auto gradNei =
                detail::exchangeProcNeighbourGradient(exec, mesh, gradPhi.internalVector());
            const auto [weightsB, corrVecB, gradNeiV] = views(
                geometryScheme->weights().boundaryData().value(),
                geometryScheme->nonOrthCorrectionVec3s().boundaryData().value(),
                gradNei
            );
            NeoN::parallelFor(
                exec,
                {0, nProcBoundaryFaces},
                NEON_LAMBDA(const localIdx procFacei) {
                    auto bcfacei = nBoundaryFaces + procFacei;
                    auto own = boundaryFaceOwners[bcfacei];
                    Vec3 ortho = nonOrthDeltaCoeffsB[bcfacei] * (phiBCValue[bcfacei] - phi[own]);
                    Tensor interpGrad = weightsB[bcfacei] * gradPhiV[own]
                                      + (scalar(1) - weightsB[bcfacei]) * gradNeiV[procFacei];
                    phifB[bcfacei] = ortho + (interpGrad & corrVecB[bcfacei]);
                },
                "computeCorrectedFaceNormalGradProcBoundaryVec3"
            );
        }
#endif
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
        // The Laplacian RHS update consumes only internal-face correction terms
        // (corrField.boundaryData() is never read). Make that contract explicit and
        // safe rather than relying on zero-init: zero the boundary so a future RHS
        // change that iterates boundary faces reads a defined value (review N5).
        NeoN::fill(corrField.boundaryData().value(), zero<ValueType>());
    }
    else if constexpr (std::is_same_v<ValueType, Vec3>)
    {
        const UnstructuredMesh& mesh = corrField.mesh();
        const auto& exec = corrField.exec();

        GaussGreenGrad gradScheme(exec, mesh);
        VolumeField<Tensor> gradPhi = gradScheme.gradTensor(volField);

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
                Tensor interpGrad = weights[facei] * gradPhiV[owners[facei]]
                                  + (scalar(1) - weights[facei]) * gradPhiV[neighbors[facei]];
                corrf[facei] = interpGrad & corrVec[facei];
            },
            "computeCorrectionTermInternalVec3"
        );
        // boundary correction not consumed by the Laplacian RHS; zero it explicitly (review N5)
        NeoN::fill(corrField.boundaryData().value(), zero<ValueType>());
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
