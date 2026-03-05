// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGradVec3.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief Compute gradient of a Vec3 field producing a Tensor field.
 *
 * For each face: compute outer product Sf (x) phif, accumulate into owner/neighbour cells.
 * Tensor component T_ij = sum_f (Sf_i * phif_j) / V
 */
void computeGradVec3(
    const VolumeField<Vec3>& in,
    const SurfaceInterpolation<Vec3>& surfInterp,
    Vector<Tensor>& out,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = in.mesh();
    const auto exec = out.exec();
    SurfaceField<Vec3> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
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

    // Internal faces: outer product Sf (x) phif
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            Vec3 Sf = faceAreaS[i];
            Vec3 phi = surfPhif[i];
            // Per-component atomic add for the outer product
            for (size_t row = 0; row < 3; row++)
            {
                for (size_t col = 0; col < 3; col++)
                {
                    scalar flux = Sf[row] * phi[col];
                    Kokkos::atomic_add(
                        &surfGradPhi[surfOwner[i]][row * 3 + col], flux
                    );
                    Kokkos::atomic_sub(
                        &surfGradPhi[surfNeighbour[i]][row * 3 + col], flux
                    );
                }
            }
        },
        "computeGradVec3Internal"
    );

    // Boundary faces
    parallelFor(
        exec,
        {nInternalFaces, surfPhif.size()},
        NEON_LAMBDA(const localIdx i) {
            auto own = surfFaceCells[i - nInternalFaces];
            Vec3 Sf = faceAreaS[i];
            Vec3 phi = surfPhif[i];
            for (size_t row = 0; row < 3; row++)
            {
                for (size_t col = 0; col < 3; col++)
                {
                    scalar flux = Sf[row] * phi[col];
                    Kokkos::atomic_add(
                        &surfGradPhi[own][row * 3 + col], flux
                    );
                }
            }
        },
        "computeGradVec3Boundary"
    );

    // Divide by cell volume
    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx celli) {
            surfGradPhi[celli] *= operatorScaling[celli] / surfV[celli];
        },
        "computeGradVec3Cells"
    );
}

GaussGreenGradVec3::GaussGreenGradVec3(const Executor& exec, const UnstructuredMesh& mesh)
    : Base(exec, mesh),
      surfaceInterpolation_(
          exec, mesh, std::make_unique<Linear<Vec3>>(exec, mesh, Dictionary())
      ) {};


void GaussGreenGradVec3::grad(
    const VolumeField<Vec3>& phi, const dsl::Coeff operatorScaling, Vector<Tensor>& gradPhi
) const
{
    computeGradVec3(phi, surfaceInterpolation_, gradPhi, operatorScaling);
};

VolumeField<Tensor>
GaussGreenGradVec3::grad(const VolumeField<Vec3>& phi, const dsl::Coeff operatorScaling) const
{
    auto gradBCs = createCalculatedBCs<VolumeBoundary<Tensor>>(phi.mesh());
    VolumeField<Tensor> gradPhi = VolumeField<Tensor>(phi.exec(), "gradPhi", phi.mesh(), gradBCs);
    fill(gradPhi.internalVector(), zero<Tensor>());
    computeGradVec3(phi, surfaceInterpolation_, gradPhi.internalVector(), operatorScaling);
    return gradPhi;
}

} // namespace NeoN
