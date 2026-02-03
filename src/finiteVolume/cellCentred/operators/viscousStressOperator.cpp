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
Vec3 fusedViscousStressFlux(
    const Vec3& Sf,
    const scalar magSf,
    const scalar nuFace,
    const scalar nutFace,
    const scalar nuTildeFace,
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
    // -nu*dev2(T(gradU)) part
    // -------------------------

    const scalar dUx_dx = gUx[0], dUx_dy = gUx[1], dUx_dz = gUx[2];
    const scalar dUy_dx = gUy[0], dUy_dy = gUy[1], dUy_dz = gUy[2];
    const scalar dUz_dx = gUz[0], dUz_dy = gUz[1], dUz_dz = gUz[2];

    const scalar divU = dUx_dx + dUy_dy + dUz_dz;

    constexpr scalar twoThird = scalar(2.0 / 3.0);

    // symm(gradU)
    // const scalar Sxy = (dUx_dy + dUy_dx);
    // const scalar Sxz = (dUx_dz + dUz_dx);
    // const scalar Syz = (dUy_dz + dUz_dy);

    // dev2(symmTensor) rows
    const Vec3 tauX {nuFace * (dUx_dx - twoThird * divU), nuFace * dUy_dx, nuFace * dUz_dx};
    const Vec3 tauY {nuFace * dUx_dy, nuFace * (dUy_dy - twoThird * divU), nuFace * dUz_dy};
    const Vec3 tauZ {nuFace * dUx_dz, nuFace * dUy_dz, nuFace * (dUz_dz - twoThird * divU)};

    // flux components = Sf · tauRow
    Vec3 t_dev;
    t_dev[0] = Sf[0] * tauX[0] + Sf[1] * tauX[1] + Sf[2] * tauX[2];
    t_dev[1] = Sf[0] * tauY[0] + Sf[1] * tauY[1] + Sf[2] * tauY[2];
    t_dev[2] = Sf[0] * tauZ[0] + Sf[1] * tauZ[1] + Sf[2] * tauZ[2];

    // -------------------------
    // Reynoldsstress part for div(R)
    // -------------------------

    const scalar chi = nuTildeFace / nuFace;
    const scalar chi3 = chi * chi * chi;

    const scalar fv1 = chi3 / (chi3 + scalar(357.911)); // pow3(Cv1) = 357.911

    const scalar kFace =
        Kokkos::cbrt(fv1) * nuTildeFace * Kokkos::sqrt(2.0 / 0.09); // Cmu = 0.09
                                                                    //* magSymmGradUFace;

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
    const SurfaceField<scalar>& nuTildeF,
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
        nuTildeF,
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
    const SurfaceField<scalar>& nuTildeF,
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
        nuTildeF,
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
    const SurfaceField<scalar>& nuTildeF,
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

    // fill(rhs, zero<Vec3>());

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

    const auto [Sf, magSf, nuFace, nutFace, nuTildeFace, dUdnF, gUx, gUy, gUz, vol, rhsV] = views(
        mesh.faceAreas(),
        mesh.magFaceAreas(),
        nuF.internalVector(),
        nutF.internalVector(),
        nuTildeF.internalVector(),
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

            const Vec3 flux = fusedViscousStressFlux(
                Sf[f],
                magSf[f],
                nuFace[f],
                nutFace[f],
                nuTildeFace[f],
                dUdnF[f],
                gUx[f],
                gUy[f],
                gUz[f]
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

            const Vec3 flux = fusedViscousStressFlux(
                Sf[f],
                magSf[f],
                nuFace[f],
                nutFace[f],
                nuTildeFace[f],
                dUdnF[f],
                gUx[f],
                gUy[f],
                gUz[f]
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
    /*    constexpr scalar twoThird = scalar(2.0 / 3.0);
        const auto [gUxV,gUyV,gUzV] =
    views(gradUx.internalVector(),gradUy.internalVector(),gradUz.internalVector()); parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c)
        {
            const Vec3 gUx2 = gUxV[c];
            const Vec3 gUy2 = gUyV[c];
            const Vec3 gUz2 = gUzV[c];

            const scalar divU = gUx2[0] + gUy2[1] + gUz2[2];

            rhsV[c] = Vec3(
                (gUx2[0] - twoThird * divU),
                gUy2[0],
                gUz2[0]
            );
        },
        "tauX_cell"
    ); */
}

} // namespace NeoN::finiteVolume::cellCentred
