// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/viscousStressOperator.hpp"

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"

namespace NeoN::finiteVolume::cellCentred
{

namespace
{

KOKKOS_INLINE_FUNCTION
Vec3 fusedViscousStressFlux(const Vec3& Sf, const Vec3& Tx, const Vec3& Ty, const Vec3& Tz)
{
    // flux components = Sf · tauRow
    Vec3 t_dev;

    t_dev[0] = Sf[0] * Tx[0] + Sf[1] * Tx[1] + Sf[2] * Tx[2];
    t_dev[1] = Sf[0] * Ty[0] + Sf[1] * Ty[1] + Sf[2] * Ty[2];
    t_dev[2] = Sf[0] * Tz[0] + Sf[1] * Tz[1] + Sf[2] * Tz[2];

    return scalar(-1.0) * t_dev;
}

KOKKOS_INLINE_FUNCTION
void computeNuEffDev2TGradU(
    const Vec3& gUx, // grad(Ux) = (dUx/dx, dUx/dy, dUx/dz)
    const Vec3& gUy, // grad(Uy)
    const Vec3& gUz, // grad(Uz)
    const scalar nu,
    const scalar nut,
    Vec3& tauX, // row X of nuEff*dev2(T(gradU))
    Vec3& tauY, // row Y
    Vec3& tauZ  // row Z
)
{
    constexpr scalar twoThird = scalar(2.0 / 3.0);

    // divergence of velocity
    const scalar divU = gUx[0] + gUy[1] + gUz[2];

    const scalar nuEff = nu + nut;

    // T(gradU) rows:
    // rowX = (dUx/dx, dUy/dx, dUz/dx)
    // rowY = (dUx/dy, dUy/dy, dUz/dy)
    // rowZ = (dUx/dz, dUy/dz, dUz/dz)

    tauX = Vec3 {nuEff * (gUx[0] - twoThird * divU), nuEff * gUy[0], nuEff * gUz[0]};

    tauY = Vec3 {nuEff * gUx[1], nuEff * (gUy[1] - twoThird * divU), nuEff * gUz[1]};

    tauZ = Vec3 {nuEff * gUx[2], nuEff * gUy[2], nuEff * (gUz[2] - twoThird * divU)};
}

KOKKOS_INLINE_FUNCTION
void atomicAddVec3(Vec3* dst, const Vec3& v)
{
    Kokkos::atomic_add(&(*dst)[0], v[0]);
    Kokkos::atomic_add(&(*dst)[1], v[1]);
    Kokkos::atomic_add(&(*dst)[2], v[2]);
}

KOKKOS_INLINE_FUNCTION
void atomicSubVec3(Vec3* dst, const Vec3& v)
{
    Kokkos::atomic_sub(&(*dst)[0], v[0]);
    Kokkos::atomic_sub(&(*dst)[1], v[1]);
    Kokkos::atomic_sub(&(*dst)[2], v[2]);
}

} // namespace

// ----------------------------
// GaussViscousStress entry point
// ----------------------------

void GaussViscousStress::explicitOp(
    Vector<Vec3>& rhs,
    const VolumeField<scalar>& nu,
    const VolumeField<scalar>& nut,
    const TensorVecField& gradU,
    const dsl::Coeff operatorScaling
) const
{
    computeViscousStressExp(surfaceInterpolationVec_, nu, nut, gradU, rhs, operatorScaling);
}

VolumeField<Vec3> GaussViscousStress::viscousStress(
    const VolumeField<scalar>& nu,
    const VolumeField<scalar>& nut,
    const TensorVecField& gradU,
    const dsl::Coeff operatorScaling
) const
{
    std::string name = "div((nuEff*dev2(T(grad(U)))))";
    VolumeField<Vec3> result(exec_, name, mesh_, createCalculatedBCs<VolumeBoundary<Vec3>>(mesh_));
    fill(result.internalVector(), zero<Vec3>());
    fill(result.boundaryData().value(), zero<Vec3>());
    computeViscousStressExp(
        surfaceInterpolationVec_, nu, nut, gradU, result.internalVector(), operatorScaling
    );
    return result;
}

// ----------------------------
// Low-level kernel implementation
// ----------------------------
void computeViscousStressExp(
    const SurfaceInterpolation<Vec3>& surfaceInterpolationVec,
    const VolumeField<scalar>& nu,
    const VolumeField<scalar>& nut,
    const TensorVecField& gradU,
    Vector<Vec3>& rhs,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = nut.mesh();
    const auto exec = nut.exec();

    auto calcVolVecBC = createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh);
    auto calcSurfVecBC = createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh);

    VolumeField<Vec3> tauX(exec, "tauX", mesh, calcVolVecBC);
    VolumeField<Vec3> tauY(exec, "tauY", mesh, calcVolVecBC);
    VolumeField<Vec3> tauZ(exec, "tauZ", mesh, calcVolVecBC);

    const auto [gUxV, gUyV, gUzV, nutV, nuV, tauXV, tauYV, tauZV] = views(
        gradU.Tx.internalVector(),
        gradU.Ty.internalVector(),
        gradU.Tz.internalVector(),
        nut.internalVector(),
        nu.internalVector(),
        tauX.internalVector(),
        tauY.internalVector(),
        tauZ.internalVector()
    );

    const localIdx nCells = static_cast<localIdx>(nut.internalVector().size());

    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx i) {
            Vec3 tauX, tauY, tauZ;

            computeNuEffDev2TGradU(gUxV[i], gUyV[i], gUzV[i], nuV[i], nutV[i], tauX, tauY, tauZ);

            tauXV[i] = tauX;
            tauYV[i] = tauY;
            tauZV[i] = tauZ;
        },
        "SA-DDES::R::internal"
    );

    const auto [gUxB, gUyB, gUzB, nutB, nuB, tauXB, tauYB, tauZB] = views(
        gradU.Tx.boundaryData().value(),
        gradU.Ty.boundaryData().value(),
        gradU.Tz.boundaryData().value(),
        nut.boundaryData().value(),
        nu.boundaryData().value(),
        tauX.boundaryData().value(),
        tauY.boundaryData().value(),
        tauZ.boundaryData().value()
    );

    const localIdx nBF = static_cast<localIdx>(tauX.boundaryData().value().size());

    parallelFor(
        exec,
        {0, nBF},
        NEON_LAMBDA(const localIdx bf) {
            Vec3 tauX, tauY, tauZ;

            computeNuEffDev2TGradU(
                gUxB[bf], gUyB[bf], gUzB[bf], nuB[bf], nutB[bf], tauX, tauY, tauZ
            );

            tauXB[bf] = tauX;
            tauYB[bf] = tauY;
            tauZB[bf] = tauZ;
        },
        "SA-DDES::R::internal"
    );

    // interpolate gradU components
    SurfaceField<Vec3> tauXF(exec, "tauXF", mesh, calcSurfVecBC);
    SurfaceField<Vec3> tauYF(exec, "tauYF", mesh, calcSurfVecBC);
    SurfaceField<Vec3> tauZF(exec, "tauZF", mesh, calcSurfVecBC);
    // fill(gUxF.internalVector(), zero<Vec3>());
    // fill(gUyF.internalVector(), zero<Vec3>());
    // fill(gUzF.internalVector(), zero<Vec3>());

    surfaceInterpolationVec.interpolate(tauX, tauXF);
    surfaceInterpolationVec.interpolate(tauY, tauYF);
    surfaceInterpolationVec.interpolate(tauZ, tauZF);

    const auto [owner, neighbour, faceCells] =
        views(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());

    const auto [Sf, tauXFV, tauYFV, tauZFV, vol, rhsV] = views(
        mesh.faceAreas(),
        tauXF.internalVector(),
        tauYF.internalVector(),
        tauZF.internalVector(),
        mesh.cellVolumes(),
        rhs
    );

    const localIdx nIF = mesh.nInternalFaces();
    const localIdx nFaces = tauXF.size();

    // -------------------------
    // Internal faces
    // -------------------------
    parallelFor(
        exec,
        {0, nIF},
        NEON_LAMBDA(const localIdx f) {
            const localIdx o = owner[f];
            const localIdx n = neighbour[f];

            const Vec3 flux = fusedViscousStressFlux(Sf[f], tauXFV[f], tauYFV[f], tauZFV[f]);

            atomicAddVec3(&rhsV[o], flux);
            atomicSubVec3(&rhsV[n], flux);
        },
        "viscousStressFused_Internal"
    );

    // -------------------------
    // Boundary faces
    // -------------------------
    parallelFor(
        exec,
        {nIF, nFaces},
        NEON_LAMBDA(const localIdx f) {
            const localIdx own = faceCells[f - nIF];

            const Vec3 flux = fusedViscousStressFlux(Sf[f], tauXFV[f], tauYFV[f], tauZFV[f]);

            atomicAddVec3(&rhsV[own], flux);
        },
        "viscousStressFused_Boundary"
    );

    // -------------------------
    // Normalize
    // -------------------------
    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { rhsV[c] *= operatorScaling[c] / vol[c]; },
        "viscousStressFused_Normalize"
    );
}

} // namespace NeoN::finiteVolume::cellCentred
