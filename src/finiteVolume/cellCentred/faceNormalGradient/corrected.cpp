// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>

#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/corrected.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/tensorVecField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

// Adds the non-orthogonal correction: corrVec & gradPhiFace for each face
void addScalarCorrection(
    const SurfaceField<Vec3>& corrVec,
    const SurfaceField<Vec3>& gradPhiFace,
    SurfaceField<scalar>& surfField
)
{
    const auto exec = surfField.exec();
    const auto [corr, gradF, result] = views(
        corrVec.internalVector(), gradPhiFace.internalVector(), surfField.internalVector()
    );

    parallelFor(
        exec,
        {0, result.size()},
        NEON_LAMBDA(const localIdx facei) { result[facei] += corr[facei] & gradF[facei]; },
        "addScalarCorrectionAllFaces"
    );
}

void addVec3Correction(
    const SurfaceField<Vec3>& corrVec,
    const SurfaceField<Vec3>& gradUxFace,
    const SurfaceField<Vec3>& gradUyFace,
    const SurfaceField<Vec3>& gradUzFace,
    SurfaceField<Vec3>& surfField
)
{
    const auto exec = surfField.exec();
    const auto [corr, gxF, gyF, gzF, result] = views(
        corrVec.internalVector(),
        gradUxFace.internalVector(),
        gradUyFace.internalVector(),
        gradUzFace.internalVector(),
        surfField.internalVector()
    );

    parallelFor(
        exec,
        {0, result.size()},
        NEON_LAMBDA(const localIdx facei) {
            result[facei] += Vec3 {
                corr[facei] & gxF[facei], corr[facei] & gyF[facei], corr[facei] & gzF[facei]};
        },
        "addVec3CorrectionAllFaces"
    );
}

// --- scalar specialization ---

void computeCorrection(
    const VolumeField<scalar>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    const GaussGreenGrad& grad,
    const SurfaceInterpolation<Vec3>& surfInterpVec3,
    SurfaceField<scalar>& correctionField
)
{
    const auto& mesh = volField.mesh();
    const auto exec = volField.exec();

    // Compute cell-centred gradient
    VolumeField<Vec3> gradPhi = grad.grad(volField);

    // Interpolate gradient to faces
    SurfaceField<Vec3> gradPhiFace(
        exec, "gradPhiFace", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
    );
    surfInterpVec3.interpolate(gradPhi, gradPhiFace);

    // correction = corrVec & gradPhiFace
    fill(correctionField.internalVector(), zero<scalar>());
    addScalarCorrection(geometryScheme->nonOrthCorrectionVec3s(), gradPhiFace, correctionField);
}

void computeCorrectedFaceNormalGrad(
    const VolumeField<scalar>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    const GaussGreenGrad& grad,
    const SurfaceInterpolation<Vec3>& surfInterpVec3,
    SurfaceField<scalar>& surfaceField
)
{
    // Uncorrected part
    computeFaceNormalGrad(volField, geometryScheme, surfaceField);

    // Compute and add correction
    const auto& mesh = volField.mesh();
    const auto exec = volField.exec();
    VolumeField<Vec3> gradPhi = grad.grad(volField);
    SurfaceField<Vec3> gradPhiFace(
        exec, "gradPhiFace", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
    );
    surfInterpVec3.interpolate(gradPhi, gradPhiFace);
    addScalarCorrection(geometryScheme->nonOrthCorrectionVec3s(), gradPhiFace, surfaceField);
}

// --- Vec3 specialization ---

void computeCorrection(
    const VolumeField<Vec3>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    const GaussGreenGrad& grad,
    const SurfaceInterpolation<Vec3>& surfInterpVec3,
    SurfaceField<Vec3>& correctionField
)
{
    const auto& mesh = volField.mesh();
    const auto exec = volField.exec();

    // Compute tensor gradient (three Vec3 rows)
    TensorVecField gradU = grad.grad(volField);

    // Interpolate each row to faces
    SurfaceField<Vec3> gradUxFace(
        exec, "gradUxFace", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
    );
    SurfaceField<Vec3> gradUyFace(
        exec, "gradUyFace", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
    );
    SurfaceField<Vec3> gradUzFace(
        exec, "gradUzFace", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
    );
    surfInterpVec3.interpolate(gradU.Tx, gradUxFace);
    surfInterpVec3.interpolate(gradU.Ty, gradUyFace);
    surfInterpVec3.interpolate(gradU.Tz, gradUzFace);

    // correction = Vec3(corrVec & gxFace, corrVec & gyFace, corrVec & gzFace)
    fill(correctionField.internalVector(), zero<Vec3>());
    addVec3Correction(
        geometryScheme->nonOrthCorrectionVec3s(), gradUxFace, gradUyFace, gradUzFace, correctionField
    );
}

void computeCorrectedFaceNormalGrad(
    const VolumeField<Vec3>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    const GaussGreenGrad& grad,
    const SurfaceInterpolation<Vec3>& surfInterpVec3,
    SurfaceField<Vec3>& surfaceField
)
{
    // Uncorrected part
    computeFaceNormalGrad(volField, geometryScheme, surfaceField);

    // Compute tensor gradient (three Vec3 rows)
    const auto& mesh = volField.mesh();
    const auto exec = volField.exec();
    TensorVecField gradU = grad.grad(volField);

    // Interpolate each row to faces
    SurfaceField<Vec3> gradUxFace(
        exec, "gradUxFace", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
    );
    SurfaceField<Vec3> gradUyFace(
        exec, "gradUyFace", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
    );
    SurfaceField<Vec3> gradUzFace(
        exec, "gradUzFace", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
    );
    surfInterpVec3.interpolate(gradU.Tx, gradUxFace);
    surfInterpVec3.interpolate(gradU.Ty, gradUyFace);
    surfInterpVec3.interpolate(gradU.Tz, gradUzFace);

    addVec3Correction(
        geometryScheme->nonOrthCorrectionVec3s(), gradUxFace, gradUyFace, gradUzFace, surfaceField
    );
}

} // namespace NeoN
