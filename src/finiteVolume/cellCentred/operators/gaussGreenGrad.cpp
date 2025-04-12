// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief free standing function implementation of the explicit gradient operator
** ie computes \sum_f \phi_f
**
** @param[in] in - Field on which the gradient should be computed
** @param[in,out] out - Field to hold the result
*/
void computeGrad(
    const VolumeField<scalar>& in,
    const SurfaceInterpolation<scalar>& surfInterp,
    VolumeField<Vector>& out
)
{
    const UnstructuredMesh& mesh = out.mesh();
    const auto exec = out.exec();
    SurfaceField<scalar> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
    );
    surfInterp.interpolate(in, phif);

    auto surfGradPhi = out.internalField().view();

    const auto [surfFaceCells, sBSf, surfPhif, surfOwner, surfNeighbour, faceAreaS, surfV] = spans(
        mesh.boundaryMesh().faceCells(),
        mesh.boundaryMesh().sf(),
        phif.internalField(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.faceAreas(),
        mesh.cellVolumes()
    );

    size_t nInternalFaces = mesh.nInternalFaces();

    // TODO use NeoN::atomic_
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t i) {
            Vector flux = faceAreaS[i] * surfPhif[i];
            Kokkos::atomic_add(&surfGradPhi[static_cast<size_t>(surfOwner[i])], flux);
            Kokkos::atomic_sub(&surfGradPhi[static_cast<size_t>(surfNeighbour[i])], flux);
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, surfPhif.size()},
        KOKKOS_LAMBDA(const size_t i) {
            size_t own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
            Vector valueOwn = faceAreaS[i] * surfPhif[i];
            Kokkos::atomic_add(&surfGradPhi[own], valueOwn);
        }
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        KOKKOS_LAMBDA(const size_t celli) { surfGradPhi[celli] *= 1 / surfV[celli]; }
    );
}

GaussGreenGrad::GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh)
    : mesh_(mesh), surfaceInterpolation_(
                       exec, mesh, std::make_unique<Linear<scalar>>(exec, mesh, Dictionary())
                   ) {};


void GaussGreenGrad::grad(const VolumeField<scalar>& phi, VolumeField<Vector>& gradPhi)
{
    computeGrad(phi, surfaceInterpolation_, gradPhi);
};

VolumeField<Vector> GaussGreenGrad::grad(const VolumeField<scalar>& phi)
{
    auto gradBCs = createCalculatedBCs<VolumeBoundary<Vector>>(phi.mesh());
    VolumeField<Vector> gradPhi = VolumeField<Vector>(phi.exec(), "gradPhi", phi.mesh(), gradBCs);
    fill(gradPhi.internalField(), zero<Vector>());
    computeGrad(phi, surfaceInterpolation_, gradPhi);
    return gradPhi;
}

} // namespace NeoN
