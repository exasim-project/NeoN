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
Vec3 divNuDev2TGradUFaceFlux(
    const Vec3& Sf,
    const scalar nuFace,
    const Vec3& gUx, // grad(Ux) = (dUx/dx, dUx/dy, dUx/dz)
    const Vec3& gUy, // grad(Uy)
    const Vec3& gUz  // grad(Uz)
)
{
    // unpack gradients (row convention!)
    const scalar dUx_dx = gUx[0], dUx_dy = gUx[1], dUx_dz = gUx[2];
    const scalar dUy_dx = gUy[0], dUy_dy = gUy[1], dUy_dz = gUy[2];
    const scalar dUz_dx = gUz[0], dUz_dy = gUz[1], dUz_dz = gUz[2];

    const scalar divU = dUx_dx + dUy_dy + dUz_dz;

    constexpr scalar twoThird = scalar(2.0 / 3.0);
    constexpr scalar half = scalar(0.5);

    // symm(gradU)
    const scalar Sxy = half * (dUx_dy + dUy_dx);
    const scalar Sxz = half * (dUx_dz + dUz_dx);
    const scalar Syz = half * (dUy_dz + dUz_dy);

    // dev2(symmTensor) rows (matches your buildTauRows)
    const Vec3 tauX {nuFace * (dUx_dx - twoThird * divU), nuFace * Sxy, nuFace * Sxz};
    const Vec3 tauY {nuFace * Sxy, nuFace * (dUy_dy - twoThird * divU), nuFace * Syz};
    const Vec3 tauZ {nuFace * Sxz, nuFace * Syz, nuFace * (dUz_dz - twoThird * divU)};

    // flux components = Sf · tauRow
    Vec3 flux;
    flux[0] = Sf[0] * tauX[0] + Sf[1] * tauX[1] + Sf[2] * tauX[2];
    flux[1] = Sf[0] * tauY[0] + Sf[1] * tauY[1] + Sf[2] * tauY[2];
    flux[2] = Sf[0] * tauZ[0] + Sf[1] * tauZ[1] + Sf[2] * tauZ[2];

    // this flux is for +div(...); your operator is -div(...):
    return scalar(-1) * flux;
}


KOKKOS_INLINE_FUNCTION
Vec3 fusedViscousStressFlux(
    const Vec3& Sf,
    const scalar magSf,
    const scalar nuFace,
    const scalar nutFace,
    const Vec3& dUdn, // snGrad(U)
    const Vec3& gUx,  // grad(Ux) at face
    const Vec3& gUy,
    const Vec3& gUz
)
{
    const scalar invMag = scalar(1) / magSf;
    const Vec3 nf = Sf * invMag;

    // -------------------------
    // Laplacian part: div(nut grad U)
    // -------------------------
    Vec3 F_lap = dUdn;
    F_lap *= (nutFace * magSf);

    // -------------------------
    // dev(2 sym gradU) part
    // -------------------------

    const scalar dUx_dx = gUx[0], dUx_dy = gUx[1], dUx_dz = gUx[2];
    const scalar dUy_dx = gUy[0], dUy_dy = gUy[1], dUy_dz = gUy[2];
    const scalar dUz_dx = gUz[0], dUz_dy = gUz[1], dUz_dz = gUz[2];

    const scalar divU = dUx_dx + dUy_dy + dUz_dz;

    constexpr scalar twoThird = scalar(2.0 / 3.0);
    constexpr scalar half = scalar(0.5);

    // symm(gradU)
    const scalar Sxy = half * (dUx_dy + dUy_dx);
    const scalar Sxz = half * (dUx_dz + dUz_dx);
    const scalar Syz = half * (dUy_dz + dUz_dy);

    // dev2(symmTensor) rows (matches your buildTauRows)
    const Vec3 tauX {nuFace * (dUx_dx - twoThird * divU), nuFace * Sxy, nuFace * Sxz};
    const Vec3 tauY {nuFace * Sxy, nuFace * (dUy_dy - twoThird * divU), nuFace * Syz};
    const Vec3 tauZ {nuFace * Sxz, nuFace * Syz, nuFace * (dUz_dz - twoThird * divU)};

    // flux components = Sf · tauRow
    Vec3 t_dev;
    t_dev[0] = Sf[0] * tauX[0] + Sf[1] * tauX[1] + Sf[2] * tauX[2];
    t_dev[1] = Sf[0] * tauY[0] + Sf[1] * tauY[1] + Sf[2] * tauY[2];
    t_dev[2] = Sf[0] * tauZ[0] + Sf[1] * tauZ[1] + Sf[2] * tauZ[2];

    return F_lap - t_dev;
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
    const SurfaceField<scalar>& nuF,
    const SurfaceField<scalar>& nutF,
    const VolumeField<Vec3>& U,
    const VolumeField<Vec3>& gradUx,
    const VolumeField<Vec3>& gradUy,
    const VolumeField<Vec3>& gradUz,
    const dsl::Coeff operatorScaling
) const
{
    computeViscousStressExp(
        faceNormalGradient_,
        surfaceInterpolationVec_,
        nuF,
        nutF,
        U,
        gradUx,
        gradUy,
        gradUz,
        rhs,
        operatorScaling
    );
}

VolumeField<Vec3> GaussViscousStress::viscousStress(
    const SurfaceField<scalar>& nuF,
    const SurfaceField<scalar>& nutF,
    const VolumeField<Vec3>& U,
    const VolumeField<Vec3>& gradUx,
    const VolumeField<Vec3>& gradUy,
    const VolumeField<Vec3>& gradUz,
    const dsl::Coeff operatorScaling
) const
{
    std::string name = "viscousStress(" + nuF.name + "," + nutF.name + "," + U.name + ")";
    VolumeField<Vec3> result(exec_, name, mesh_, createCalculatedBCs<VolumeBoundary<Vec3>>(mesh_));
    fill(result.internalVector(), zero<Vec3>());
    fill(result.boundaryData().value(), zero<Vec3>());
    computeViscousStressExp(
        faceNormalGradient_,
        surfaceInterpolationVec_,
        nuF,
        nutF,
        U,
        gradUx,
        gradUy,
        gradUz,
        result.internalVector(),
        operatorScaling
    );
    return result;
}

// ----------------------------
// Low-level kernel implementation
// ----------------------------
void computeViscousStressExp(
    const FaceNormalGradient<Vec3>& faceNormalGradient,
    const SurfaceInterpolation<Vec3>& surfaceInterpolationVec,
    const SurfaceField<scalar>& nuF,
    const SurfaceField<scalar>& nutF,
    const VolumeField<Vec3>& U,
    const VolumeField<Vec3>& gradUx,
    const VolumeField<Vec3>& gradUy,
    const VolumeField<Vec3>& gradUz,
    Vector<Vec3>& rhs,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = U.mesh();
    const auto exec = U.exec();

    fill(rhs, zero<Vec3>());

    // snGrad(U)
    SurfaceField<Vec3> dUdn = faceNormalGradient.faceNormalGrad(U);

    // interpolate gradU components
    SurfaceField<Vec3> gUxF(exec, "gUxF", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh));
    SurfaceField<Vec3> gUyF(exec, "gUyF", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh));
    SurfaceField<Vec3> gUzF(exec, "gUzF", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh));

    surfaceInterpolationVec.interpolate(gradUx, gUxF);
    surfaceInterpolationVec.interpolate(gradUy, gUyF);
    surfaceInterpolationVec.interpolate(gradUz, gUzF);

    const auto [owner, neighbour, faceCells] =
        views(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());

    const auto [Sf, magSf, nuFace, nutFace, dUdnF, gUx, gUy, gUz, vol, rhsV] = views(
        mesh.faceAreas(),
        mesh.magFaceAreas(),
        nuF.internalVector(),
        nutF.internalVector(),
        dUdn.internalVector(),
        gUxF.internalVector(),
        gUyF.internalVector(),
        gUzF.internalVector(),
        mesh.cellVolumes(),
        rhs
    );

    const localIdx nIF = mesh.nInternalFaces();
    const localIdx nFaces = dUdnF.size();

    // -------------------------
    // Internal faces
    // -------------------------
    parallelFor(
        exec,
        {0, nIF},
        NEON_LAMBDA(const localIdx f) {
            const localIdx o = owner[f];
            const localIdx n = neighbour[f];

            //	    const Vec3 flux = divNuDev2TGradUFaceFlux(Sf[f], nuFace[f], gUx[f], gUy[f],
            // gUz[f]);
            const Vec3 flux = fusedViscousStressFlux(
                Sf[f], magSf[f], nuFace[f], nutFace[f], dUdnF[f], gUx[f], gUy[f], gUz[f]
            );

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

            //	    const Vec3 flux = divNuDev2TGradUFaceFlux(Sf[f], nuFace[f], gUx[f], gUy[f],
            // gUz[f]);
            const Vec3 flux = fusedViscousStressFlux(
                Sf[f], magSf[f], nuFace[f], nutFace[f], dUdnF[f], gUx[f], gUy[f], gUz[f]
            );

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
