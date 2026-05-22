// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>

#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeFaceNormalGrad(
    const VolumeField<ValueType>& volVector,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    SurfaceField<ValueType>& surfaceVector
)
{
    const UnstructuredMesh& mesh = surfaceVector.mesh();
    const auto& exec = surfaceVector.exec();

    const auto [owners, neighbors, boundaryFaceOwners] =
        views(mesh.faceOwners(), mesh.faceNeighbors(), mesh.boundaryMesh().faceOwners());

    const auto [phif, phifB, phi, phiBCValue, nonOrthDeltaCoeffs, nonOrthDeltaCoeffsB] = views(
        surfaceVector.internalVector(),
        surfaceVector.boundaryData().value(),
        volVector.internalVector(),
        volVector.boundaryData().value(),
        geometryScheme->nonOrthDeltaCoeffs().internalVector(),
        geometryScheme->nonOrthDeltaCoeffs().boundaryData().value()
    );

    auto nInternalFaces = mesh.nInternalFaces();
    auto nBoundaryFaces = mesh.nBoundaryFaces();

    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            phif[facei] = nonOrthDeltaCoeffs[facei] * (phi[neighbors[facei]] - phi[owners[facei]]);
        },
        "computeFaceNormalGradInternal"
    );

    NeoN::parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = boundaryFaceOwners[bfi];
            phifB[bfi] = nonOrthDeltaCoeffsB[bfi] * (phiBCValue[bfi] - phi[own]);
        },
        "computeFaceNormalGradBoundary"
    );
}

#define NF_DECLARE_COMPUTE_IMP_FNG(TYPENAME)                                                       \
    template void computeFaceNormalGrad<                                                           \
        TYPENAME>(const VolumeField<TYPENAME>&, const std::shared_ptr<GeometryScheme>, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_FNG(scalar);
NF_DECLARE_COMPUTE_IMP_FNG(Vec3);

} // namespace NeoN
