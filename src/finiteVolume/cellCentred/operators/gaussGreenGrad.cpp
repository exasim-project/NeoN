// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief free standing function implementation of the explicit gradient operator
** ie computes \sum_f \phi_f
**
** @param[in] in - Vector on which the gradient should be computed
** @param[in,out] out - Vector to hold the result
*/
void computeGrad(
    const VolumeField<scalar>& in,
    const SurfaceInterpolation<scalar>& surfInterp,
    VolumeField<Vec3>& out
)
{
    const UnstructuredMesh& mesh = out.mesh();
    const auto exec = out.exec();
    SurfaceField<scalar> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
    );
    surfInterp.interpolate(in, phif);

    auto surfGradPhi = out.internalVector().view();

    const auto [surfFaceCells, sBSf, surfPhif, surfOwner, surfNeighbour, faceAreaS, surfV] = views(
        mesh.boundaryMesh().faceCells(),
        mesh.boundaryMesh().sf(),
        phif.internalVector(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.faceAreas(),
        mesh.cellVolumes()
    );

    auto nInternalFaces = mesh.nInternalFaces();

    // TODO use NeoN::atomic_
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx i) {
            Vec3 flux = faceAreaS[i] * surfPhif[i];
            Kokkos::atomic_add(&surfGradPhi[surfOwner[i]], flux);
            Kokkos::atomic_sub(&surfGradPhi[surfNeighbour[i]], flux);
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, surfPhif.size()},
        KOKKOS_LAMBDA(const localIdx i) {
            auto own = surfFaceCells[i - nInternalFaces];
            Vec3 valueOwn = faceAreaS[i] * surfPhif[i];
            Kokkos::atomic_add(&surfGradPhi[own], valueOwn);
        }
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        KOKKOS_LAMBDA(const localIdx celli) { surfGradPhi[celli] *= 1 / surfV[celli]; }
    );
}

GaussGreenGrad::GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh)
    : mesh_(mesh), surfaceInterpolation_(
                       exec, mesh, std::make_unique<Linear<scalar>>(exec, mesh, Dictionary())
                   ) {};


void GaussGreenGrad::grad(const VolumeField<scalar>& phi, VolumeField<Vec3>& gradPhi)
{
    computeGrad(phi, surfaceInterpolation_, gradPhi);
};

VolumeField<Vec3> GaussGreenGrad::grad(const VolumeField<scalar>& phi)
{
    auto gradBCs = createCalculatedBCs<VolumeBoundary<Vec3>>(phi.mesh());
    VolumeField<Vec3> gradPhi = VolumeField<Vec3>(phi.exec(), "gradPhi", phi.mesh(), gradBCs);
    fill(gradPhi.internalVector(), zero<Vec3>());
    computeGrad(phi, surfaceInterpolation_, gradPhi);
    return gradPhi;
}

} // namespace NeoN
