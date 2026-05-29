// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>
#include <type_traits>

#include "NeoN/core/error.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/limitedCorrected.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLimitedCorrectedFaceNormalGrad(
    const VolumeField<ValueType>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    scalar limitCoeff,
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
        const scalar lc = limitCoeff;
        const scalar oneMinusLc = scalar(1) - lc;

        NeoN::parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx facei) {
                scalar ortho =
                    nonOrthDeltaCoeffs[facei] * (phi[neighbors[facei]] - phi[owners[facei]]);
                Vec3 interpGrad = weights[facei] * gradPhiV[owners[facei]]
                                + (scalar(1) - weights[facei]) * gradPhiV[neighbors[facei]];
                scalar corr = corrVec[facei] & interpGrad;

                // Limiter: bounds the correction relative to the orthogonal part
                scalar absCorr = std::abs(corr);
                scalar limiter =
                    (absCorr > scalar(0)) ? std::min(
                        lc * std::abs(ortho) / (oneMinusLc * absCorr + ROOTVSMALL), scalar(1)
                    )
                                          : scalar(1);

                phif[facei] = ortho + limiter * corr;
            },
            "computeLimitedCorrectedFaceNormalGradInternal"
        );

        // corrVec is zero at boundaries — boundary snGrad reduces to the uncorrected form.
        NeoN::parallelFor(
            exec,
            {0, nBoundaryFaces},
            NEON_LAMBDA(const localIdx bfi) {
                auto own = boundaryFaceOwners[bfi];
                phifB[bfi] = nonOrthDeltaCoeffsB[bfi] * (phiBCValue[bfi] - phi[own]);
            },
            "computeLimitedCorrectedFaceNormalGradBoundary"
        );

        auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
        if (nProcBoundaryFaces > 0)
        {
            NeoN::parallelFor(
                exec,
                {0, nProcBoundaryFaces},
                NEON_LAMBDA(const localIdx procFacei) {
                    auto bcfacei = nBoundaryFaces + procFacei;
                    auto own = boundaryFaceOwners[bcfacei];
                    phifB[bcfacei] =
                        nonOrthDeltaCoeffsB[bcfacei] * (phiBCValue[bcfacei] - phi[own]);
                },
                "computeLimitedCorrectedFaceNormalGradProcBoundary"
            );
        }
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
        const scalar lc = limitCoeff;
        const scalar oneMinusLc = scalar(1) - lc;

        NeoN::parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx facei) {
                Vec3 ortho =
                    nonOrthDeltaCoeffs[facei] * (phi[neighbors[facei]] - phi[owners[facei]]);
                Tensor interpGrad = weights[facei] * gradPhiV[owners[facei]]
                                  + (scalar(1) - weights[facei]) * gradPhiV[neighbors[facei]];
                Vec3 corr = interpGrad & corrVec[facei];

                // Limiter on the magnitudes — bounds the correction relative to the orthogonal part
                scalar absCorr = mag(corr);
                scalar limiter =
                    (absCorr > scalar(0))
                        ? std::min(lc * mag(ortho) / (oneMinusLc * absCorr + ROOTVSMALL), scalar(1))
                        : scalar(1);

                phif[facei] = ortho + limiter * corr;
            },
            "computeLimitedCorrectedFaceNormalGradInternalVec3"
        );

        NeoN::parallelFor(
            exec,
            {0, nBoundaryFaces},
            NEON_LAMBDA(const localIdx bfi) {
                auto own = boundaryFaceOwners[bfi];
                phifB[bfi] = nonOrthDeltaCoeffsB[bfi] * (phiBCValue[bfi] - phi[own]);
            },
            "computeLimitedCorrectedFaceNormalGradBoundaryVec3"
        );

        // Processor-boundary parity with the scalar path (review N3). Orthogonal
        // part only; non-orth proc correction deferred (review N4). The limiter
        // applies only to the internal correction, so proc faces reduce to the
        // uncorrected form exactly as in the scalar path.
        auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
        if (nProcBoundaryFaces > 0)
        {
            NeoN::parallelFor(
                exec,
                {0, nProcBoundaryFaces},
                NEON_LAMBDA(const localIdx procFacei) {
                    auto bcfacei = nBoundaryFaces + procFacei;
                    auto own = boundaryFaceOwners[bcfacei];
                    phifB[bcfacei] =
                        nonOrthDeltaCoeffsB[bcfacei] * (phiBCValue[bcfacei] - phi[own]);
                },
                "computeLimitedCorrectedFaceNormalGradProcBoundaryVec3"
            );
        }
    }
}

#define NF_DECLARE_COMPUTE_LIMITED_CORRECTED_FNG(TYPENAME)                                         \
    template void computeLimitedCorrectedFaceNormalGrad<                                           \
        TYPENAME>(const VolumeField<TYPENAME>&, const std::shared_ptr<GeometryScheme>, scalar, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_LIMITED_CORRECTED_FNG(scalar);
NF_DECLARE_COMPUTE_LIMITED_CORRECTED_FNG(Vec3);

template<typename ValueType>
void computeLimitedCorrectionTerm(
    const VolumeField<ValueType>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    scalar limitCoeff,
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

        const auto [corrf, phi, nonOrthDeltaCoeffs, weights, corrVec] = views(
            corrField.internalVector(),
            volField.internalVector(),
            geometryScheme->nonOrthDeltaCoeffs().internalVector(),
            geometryScheme->weights().internalVector(),
            geometryScheme->nonOrthCorrectionVec3s().internalVector()
        );

        const auto gradPhiV = gradPhi.internalVector().view();
        auto nInternalFaces = mesh.nInternalFaces();
        const scalar lc = limitCoeff;
        const scalar oneMinusLc = scalar(1) - lc;

        NeoN::parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx facei) {
                scalar ortho =
                    nonOrthDeltaCoeffs[facei] * (phi[neighbors[facei]] - phi[owners[facei]]);
                Vec3 interpGrad = weights[facei] * gradPhiV[owners[facei]]
                                + (scalar(1) - weights[facei]) * gradPhiV[neighbors[facei]];
                scalar corr = corrVec[facei] & interpGrad;

                scalar absCorr = std::abs(corr);
                scalar limiter =
                    (absCorr > scalar(0)) ? std::min(
                        lc * std::abs(ortho) / (oneMinusLc * absCorr + ROOTVSMALL), scalar(1)
                    )
                                          : scalar(1);

                corrf[facei] = limiter * corr;
            },
            "computeLimitedCorrectionTermInternal"
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

        const auto [corrf, phi, nonOrthDeltaCoeffs, weights, corrVec] = views(
            corrField.internalVector(),
            volField.internalVector(),
            geometryScheme->nonOrthDeltaCoeffs().internalVector(),
            geometryScheme->weights().internalVector(),
            geometryScheme->nonOrthCorrectionVec3s().internalVector()
        );

        const auto gradPhiV = gradPhi.internalVector().view();
        auto nInternalFaces = mesh.nInternalFaces();
        const scalar lc = limitCoeff;
        const scalar oneMinusLc = scalar(1) - lc;

        NeoN::parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx facei) {
                Vec3 ortho =
                    nonOrthDeltaCoeffs[facei] * (phi[neighbors[facei]] - phi[owners[facei]]);
                Tensor interpGrad = weights[facei] * gradPhiV[owners[facei]]
                                  + (scalar(1) - weights[facei]) * gradPhiV[neighbors[facei]];
                Vec3 corr = interpGrad & corrVec[facei];

                scalar absCorr = mag(corr);
                scalar limiter =
                    (absCorr > scalar(0))
                        ? std::min(lc * mag(ortho) / (oneMinusLc * absCorr + ROOTVSMALL), scalar(1))
                        : scalar(1);

                corrf[facei] = limiter * corr;
            },
            "computeLimitedCorrectionTermInternalVec3"
        );
        // boundary correction not consumed by the Laplacian RHS; zero it explicitly (review N5)
        NeoN::fill(corrField.boundaryData().value(), zero<ValueType>());
    }
}

#define NF_DECLARE_COMPUTE_LIMITED_CORRECTION_TERM(TYPENAME)                                       \
    template void computeLimitedCorrectionTerm<                                                    \
        TYPENAME>(const VolumeField<TYPENAME>&, const std::shared_ptr<GeometryScheme>, scalar, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_LIMITED_CORRECTION_TERM(scalar);
NF_DECLARE_COMPUTE_LIMITED_CORRECTION_TERM(Vec3);

} // namespace NeoN
