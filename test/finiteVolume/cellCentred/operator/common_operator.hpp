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
    auto faceFlux = fvcc::SurfaceVector<scalar>(exec, "sf", mesh, surfaceBCs);
    fill(faceFlux.internalVector(), 1.0);
    auto boundFaceFlux = faceFlux.internalVector().view();
    // face on the left side has different orientation
    parallelFor(
        exec,
        {mesh.nCells() - 1, mesh.nCells()},
        KOKKOS_LAMBDA(const localIdx i) { boundFaceFlux[i] = -1.0; }
    );

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);

    auto phi = fvcc::VolumeVector<TestType>(exec, "sf", mesh, volumeBCs);
    fill(phi.internalVector(), one<TestType>());
    fill(phi.boundaryData().value(), one<TestType>());
    phi.correctBoundaryConditions();

    auto result = Vector<TestType>(exec, phi.size(), zero<TestType>());

    return std::make_tuple(phi, faceFlux, result);
}
}
