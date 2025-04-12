// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include <memory>

#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeFaceNormalGrad(
    const VolumeField<ValueType>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    SurfaceField<ValueType>& surfaceField
)
{
    const UnstructuredMesh& mesh = surfaceField.mesh();
    const auto& exec = surfaceField.exec();

    const auto [owner, neighbour, surfFaceCells] =
        spans(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());


    const auto [phif, phi, phiBCValue, nonOrthDeltaCoeffs] = spans(
        surfaceField.internalField(),
        volField.internalField(),
        volField.boundaryField().value(),
        geometryScheme->nonOrthDeltaCoeffs().internalField()
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
NF_DECLARE_COMPUTE_IMP_FNG(Vector);

} // namespace NeoN
