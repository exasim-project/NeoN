// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#include "NeoN/NeoN.hpp"

namespace NeoN
{

template<typename TestType>
auto setup_operator_test(const UnstructuredMesh& mesh)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    auto exec = mesh.exec();
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);

    // compute corresponding uniform faceFlux
    // TODO this should be handled outside of the unit test
    auto faceFlux = fvcc::SurfaceField<scalar>(exec, "sf", mesh, surfaceBCs);
    fill(faceFlux.internalField(), 1.0);
    auto boundFaceFlux = faceFlux.internalField().view();
    // face on the left side has different orientation
    parallelFor(
        exec,
        {mesh.nCells() - 1, mesh.nCells()},
        KOKKOS_LAMBDA(const size_t i) { boundFaceFlux[i] = -1.0; }
    );

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);

    auto phi = fvcc::VolumeField<TestType>(exec, "sf", mesh, volumeBCs);
    fill(phi.internalField(), one<TestType>());
    fill(phi.boundaryField().value(), one<TestType>());
    phi.correctBoundaryConditions();

    auto result = Field<TestType>(exec, phi.size(), zero<TestType>());

    return std::make_tuple(phi, faceFlux, result);
}
}
