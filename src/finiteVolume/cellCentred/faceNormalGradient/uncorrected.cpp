// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
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

    const auto [owner, neighbour, surfFaceCells] =
        views(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());


    const auto [phif, phi, phiBCValue, nonOrthDeltaCoeffs] = views(
        surfaceVector.internalVector(),
        volVector.internalVector(),
        volVector.boundaryData().value(),
        geometryScheme->nonOrthDeltaCoeffs().internalVector()
    );

    auto nInternalFaces = mesh.nInternalFaces();

    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            phif[facei] = nonOrthDeltaCoeffs[facei] * (phi[neighbour[facei]] - phi[owner[facei]]);
        },
        "computeFaceNormalGradInternal"
    );

    NeoN::parallelFor(
        exec,
        {nInternalFaces, phif.size()},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto faceBCI = facei - nInternalFaces;
            auto own = surfFaceCells[faceBCI];

            phif[facei] = nonOrthDeltaCoeffs[facei] * (phiBCValue[faceBCI] - phi[own]);
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
