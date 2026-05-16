// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDivTensor.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

void computeDivTensor(
    const VolumeField<Tensor>& in,
    const SurfaceInterpolation<Tensor>& surfInterp,
    Vector<Vec3>& out,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = in.mesh();
    const auto exec = out.exec();
    SurfaceField<Tensor> Tf(exec, "Tf", mesh, createCalculatedBCs<SurfaceBoundary<Tensor>>(mesh));
    surfInterp.interpolate(in, Tf);

    auto nInternalFaces = mesh.nInternalFaces();

    // Extrapolate boundary face values from adjacent cell centers.
    // Arithmetic-created fields have zero boundary data (calculatedBCs),
    // so we use the owner cell value instead (equivalent to zeroGradient).
    {
        const auto [inInternal, faceCells] =
            views(in.internalVector(), mesh.boundaryMesh().faceCells());
        auto bndTf = Tf.boundaryData().value().view();
        auto surfTfView = Tf.internalVector().view();
        parallelFor(
            exec,
            {0, mesh.boundaryMesh().offset().back()},
            NEON_LAMBDA(const localIdx bfacei) {
                auto cellValue = inInternal[faceCells[bfacei]];
                bndTf[bfacei] = cellValue;
                surfTfView[nInternalFaces + bfacei] = cellValue;
            },
            "extrapolateBoundaryTensor"
        );
    }

    auto res = out.view();

    const auto [surfFaceCells, surfTf, surfOwner, surfNeighbour, faceAreaS, surfV] = views(
        mesh.boundaryMesh().faceCells(),
        Tf.internalVector(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.faceAreas(),
        mesh.cellVolumes()
    );

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            Vec3 flux = inner(faceAreaS[i], surfTf[i]);
            Kokkos::atomic_add(&res[surfOwner[i]], flux);
            Kokkos::atomic_sub(&res[surfNeighbour[i]], flux);
        },
        "computeDivTensorInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, surfTf.size()},
        NEON_LAMBDA(const localIdx i) {
            auto own = surfFaceCells[i - nInternalFaces];
            Vec3 flux = inner(faceAreaS[i], surfTf[i]);
            Kokkos::atomic_add(&res[own], flux);
        },
        "computeDivTensorBoundary"
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx celli) { res[celli] *= operatorScaling[celli] / surfV[celli]; },
        "computeDivTensorNormalize"
    );
}

GaussGreenDivTensor::GaussGreenDivTensor(const Executor& exec, const UnstructuredMesh& mesh)
    : exec_(exec), mesh_(mesh),
      surfaceInterpolation_(
          exec, mesh, std::make_unique<Linear<Tensor>>(exec, mesh, Dictionary())
      ) {};


VolumeField<Vec3>
GaussGreenDivTensor::div(const VolumeField<Tensor>& T, const dsl::Coeff operatorScaling) const
{
    auto divBCs = createCalculatedBCs<VolumeBoundary<Vec3>>(T.mesh());
    VolumeField<Vec3> divT = VolumeField<Vec3>(T.exec(), "divT", T.mesh(), divBCs);
    fill(divT.internalVector(), zero<Vec3>());
    computeDivTensor(T, surfaceInterpolation_, divT.internalVector(), operatorScaling);
    return divT;
}

} // namespace NeoN::finiteVolume::cellCentred
