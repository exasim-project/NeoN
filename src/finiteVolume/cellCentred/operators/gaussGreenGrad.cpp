// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
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
    Vector<Vec3>& out,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = in.mesh();
    const auto exec = out.exec();
    SurfaceField<scalar> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
    );
    surfInterp.interpolate(in, phif);

    auto surfGradPhi = out.view();

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

    // Green-Gauss gradient theorem: ∇φ_C = (1/V_C) * sum_f S_f * φ_f
    //
    // S_f points from owner to neighbour by construction (valid for all internal faces).
    //   owner cell:     S_f is the outward area vector  →  +S_f * φ_f  (add)
    //   neighbour cell: S_f points inward to neighbour  → −S_f * φ_f  (subtract)
    // TODO use NeoN::atomic_
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            Vec3 flux = faceAreaS[i] * surfPhif[i];
            Kokkos::atomic_add(&surfGradPhi[surfOwner[i]], flux);     // +S_f * φ_f
            Kokkos::atomic_sub(&surfGradPhi[surfNeighbour[i]], flux); // −S_f * φ_f
        },
        "computeGradInternal"
    );

    // Boundary faces: only the owner cell is on this rank.
    parallelFor(
        exec,
        {nInternalFaces, surfPhif.size()},
        NEON_LAMBDA(const localIdx i) {
            auto own = surfFaceCells[i - nInternalFaces];
            Vec3 valueOwn = faceAreaS[i] * surfPhif[i]; // +S_f * φ_f (S_f outward from owner)
            Kokkos::atomic_add(&surfGradPhi[own], valueOwn);
        },
        "computeGradBoundary"
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx celli) {
            surfGradPhi[celli] *= operatorScaling[celli] / surfV[celli];
        },
        "computeGradCells"
    );
}

GaussGreenGrad::GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh)
    : Base(exec, mesh), surfaceInterpolation_(
                            exec, mesh, std::make_unique<Linear<scalar>>(exec, mesh, Dictionary())
                        ) {};


void GaussGreenGrad::grad(
    const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling, Vector<Vec3>& gradPhi
) const
{
    computeGrad(phi, surfaceInterpolation_, gradPhi, operatorScaling);
};

VolumeField<Vec3>
GaussGreenGrad::grad(const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling) const
{
    auto gradBCs = createCalculatedBCs<VolumeBoundary<Vec3>>(phi.mesh());
    VolumeField<Vec3> gradPhi = VolumeField<Vec3>(phi.exec(), "gradPhi", phi.mesh(), gradBCs);
    fill(gradPhi.internalVector(), zero<Vec3>());
    computeGrad(phi, surfaceInterpolation_, gradPhi.internalVector(), operatorScaling);
    return gradPhi;
}

} // namespace NeoN
