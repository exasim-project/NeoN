// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

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
        spans(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());


    const auto [phif, phi, phiBCValue, nonOrthDeltaCoeffs] = spans(
        surfaceVector.internalVector(),
        volVector.internalVector(),
        volVector.boundaryVector().value(),
        geometryScheme->nonOrthDeltaCoeffs().internalVector()
    );

    size_t nInternalFaces = mesh.nInternalFaces();

    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            phif[facei] = nonOrthDeltaCoeffs[facei] * (phi[neighbour[facei]] - phi[owner[facei]]);
        }
    );

    NeoN::parallelFor(
        exec,
        {nInternalFaces, phif.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            auto faceBCI = facei - nInternalFaces;
            auto own = static_cast<size_t>(surfFaceCells[faceBCI]);

            phif[facei] = nonOrthDeltaCoeffs[facei] * (phiBCValue[faceBCI] - phi[own]);
        }
    );
}

#define NF_DECLARE_COMPUTE_IMP_FNG(TYPENAME)                                                       \
    template void computeFaceNormalGrad<                                                           \
        TYPENAME>(const VolumeField<TYPENAME>&, const std::shared_ptr<GeometryScheme>, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_FNG(scalar);
NF_DECLARE_COMPUTE_IMP_FNG(Vec3);

} // namespace NeoN
