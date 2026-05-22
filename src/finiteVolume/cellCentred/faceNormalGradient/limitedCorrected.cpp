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
    }
    else
    {
        NF_ERROR_EXIT("limitedCorrected snGrad for Vec3 fields is not implemented: "
                      "requires tensor gradient support");
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
        // boundary correction is intentionally not written: the laplacian RHS update
        // iterates only internal faces, so corrField.boundaryData() is never read.
    }
    else
    {
        NF_ERROR_EXIT("implicitCorrection for Vec3 fields is not implemented: "
                      "requires tensor gradient support");
    }
}

#define NF_DECLARE_COMPUTE_LIMITED_CORRECTION_TERM(TYPENAME)                                       \
    template void computeLimitedCorrectionTerm<                                                    \
        TYPENAME>(const VolumeField<TYPENAME>&, const std::shared_ptr<GeometryScheme>, scalar, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_LIMITED_CORRECTION_TERM(scalar);
NF_DECLARE_COMPUTE_LIMITED_CORRECTION_TERM(Vec3);

} // namespace NeoN
