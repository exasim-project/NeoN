// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


void computeGrad(
    const VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp,
    VolumeField<Vector>& gradPhi,
    SurfaceField<scalar>& phif
)
{
    const UnstructuredMesh& mesh = gradPhi.mesh();
    const auto exec = gradPhi.exec();
    const auto surfFaceCells = mesh.boundaryMesh().faceCells().span();
    const auto sBSf = mesh.boundaryMesh().sf().span();
    auto surfGradPhi = gradPhi.internalField().span();

    surfInterp.interpolate(phi, phif);
    const auto surfPhif = phif.internalField().span();
    const auto surfOwner = mesh.faceOwner().span();
    const auto surfNeighbour = mesh.faceNeighbour().span();
    const auto sSf = mesh.faceAreas().span();
    size_t nInternalFaces = mesh.nInternalFaces();

    const auto surfV = mesh.cellVolumes().span();

    ThreadSafeAdd add {exec};
    ThreadSafeSub sub {exec};

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t i) {
            Vector flux = sSf[i] * surfPhif[i];
            add(surfGradPhi[static_cast<size_t>(surfOwner[i])], flux);
            sub(surfGradPhi[static_cast<size_t>(surfNeighbour[i])], flux);
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, surfPhif.size()},
        KOKKOS_LAMBDA(const size_t i) {
            size_t own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
            Vector valueOwn = sSf[i] * surfPhif[i];
            add(surfGradPhi[own], valueOwn);
        }
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        KOKKOS_LAMBDA(const size_t celli) { surfGradPhi[celli] *= 1 / surfV[celli]; }
    );
}

void computeGrad(
    const VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp,
    VolumeField<Vector>& gradPhi
)
{
    const auto exec = gradPhi.exec();
    const UnstructuredMesh& mesh = gradPhi.mesh();
    SurfaceField<scalar> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
    );

    computeGrad(phi, surfInterp, gradPhi, phif);
}

GaussGreenGrad::GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh)
    : mesh_(mesh),
      surfaceInterpolation_(exec, mesh, std::make_unique<Linear>(exec, mesh, Dictionary())) {};


void GaussGreenGrad::grad(const VolumeField<scalar>& phi, VolumeField<Vector>& gradPhi)
{
    computeGrad(phi, surfaceInterpolation_, gradPhi);
};

} // namespace NeoFOAM
