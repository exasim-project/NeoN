// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/divTensorExp.hpp"

namespace NeoN::finiteVolume::cellCentred
{

KOKKOS_INLINE_FUNCTION
NeoN::scalar dot3(const NeoN::Vec3& a, const NeoN::Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

KOKKOS_INLINE_FUNCTION
void buildTauRows(
    const scalar nu,
    const Vec3& gUx,
    const Vec3& gUy,
    const Vec3& gUz,
    Vec3& tauX,
    Vec3& tauY,
    Vec3& tauZ
)
{
    const scalar dUx_dx = gUx[0];
    const scalar dUx_dy = gUx[1];
    const scalar dUx_dz = gUx[2];

    const scalar dUy_dx = gUy[0];
    const scalar dUy_dy = gUy[1];
    const scalar dUy_dz = gUy[2];

    const scalar dUz_dx = gUz[0];
    const scalar dUz_dy = gUz[1];
    const scalar dUz_dz = gUz[2];

    const scalar divU = dUx_dx + dUy_dy + dUz_dz;

    constexpr scalar twoThird = scalar(2.0 / 3.0);
    constexpr scalar half = scalar(0.5);

    const scalar Sxy = half * (dUx_dy + dUy_dx);
    const scalar Sxz = half * (dUx_dz + dUz_dx);
    const scalar Syz = half * (dUy_dz + dUz_dy);

    const scalar Sxx = dUx_dx;
    const scalar Syy = dUy_dy;
    const scalar Szz = dUz_dz;

    tauX = Vec3 {nu * (Sxx - twoThird * divU), nu * Sxy, nu * Sxz};
    tauY = Vec3 {nu * Sxy, nu * (Syy - twoThird * divU), nu * Syz};
    tauZ = Vec3 {nu * Sxz, nu * Syz, nu * (Szz - twoThird * divU)};
}


void computeDivVectorExp(
    const VolumeField<Vec3>& a,                      // volume vector field (row of tensor)
    const SurfaceInterpolation<Vec3>& surfInterpVec, // face interpolation for Vec3
    Vector<scalar>& divA,                            // cell scalar result
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = a.mesh();
    const auto exec = a.exec();

    // Interpolate a -> faces
    SurfaceField<Vec3> aF(exec, "aF", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh));
    surfInterpVec.interpolate(a, aF);

    const localIdx nIF = mesh.nInternalFaces();
    const localIdx nBF = mesh.nBoundaryFaces();

    const auto [owner, neighbour, faceCells] =
        views(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());

    const auto [SfAll, vol] = views(mesh.faceAreas(), mesh.cellVolumes());

    auto res = divA.view();

    {
        auto aAll = aF.internalVector().view();
        NF_DEBUG_ASSERT(
            aAll.size() == nIF + nBF, "surface field size must be nInternalFaces + nBoundaryFaces"
        );

        parallelFor(
            exec,
            {0, nIF},
            NEON_LAMBDA(const localIdx f) {
                const localIdx o = owner[f];
                const localIdx n = neighbour[f];

                const scalar flux = dot3(SfAll[f], aAll[f]); // Sf · a_f

                Kokkos::atomic_add(&res[o], flux);
                Kokkos::atomic_sub(&res[n], flux);
            },
            "divVectorExp_Internal"
        );
    }

    // ---- Boundary faces (owner only)
    {
        auto aAll = aF.internalVector().view();

        parallelFor(
            exec,
            {nIF, nIF + nBF},
            NEON_LAMBDA(const localIdx f) {
                const localIdx own = faceCells[f - nIF];

                const scalar flux = dot3(SfAll[f], aAll[f]);

                Kokkos::atomic_add(&res[own], flux);
            },
            "divVectorExp_Boundary"
        );
    }

    // ---- Normalize by volume and operator scaling
    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { res[c] *= operatorScaling[c] / vol[c]; },
        "divVectorExp_Normalize"
    );
}

void computeDivNuDev2TGradUExp(
    const VolumeField<scalar>& nu,
    const VolumeField<Vec3>& gradUx,
    const VolumeField<Vec3>& gradUy,
    const VolumeField<Vec3>& gradUz,
    const SurfaceInterpolation<Vec3>& surfInterpVec,
    Vector<Vec3>& rhs,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = nu.mesh();
    const auto exec = nu.exec();

    fill(rhs, zero<Vec3>());

    auto calcBCs = createCalculatedBCs<VolumeBoundary<Vec3>>(mesh);

    VolumeField<Vec3> tauX(exec, "tauX", mesh, calcBCs);
    VolumeField<Vec3> tauY(exec, "tauY", mesh, calcBCs);
    VolumeField<Vec3> tauZ(exec, "tauZ", mesh, calcBCs);

    Vector<scalar> divTauX(exec, mesh.nCells());
    Vector<scalar> divTauY(exec, mesh.nCells());
    Vector<scalar> divTauZ(exec, mesh.nCells());

    fill(divTauX, scalar(0));
    fill(divTauY, scalar(0));
    fill(divTauZ, scalar(0));

    // -------------------------
    // Internal cells
    // -------------------------
    {
        const auto [nuV, gUx, gUy, gUz, tX, tY, tZ] = views(
            nu.internalVector(),
            gradUx.internalVector(),
            gradUy.internalVector(),
            gradUz.internalVector(),
            tauX.internalVector(),
            tauY.internalVector(),
            tauZ.internalVector()
        );

        parallelFor(
            exec,
            {0, mesh.nCells()},
            NEON_LAMBDA(const localIdx c) {
                buildTauRows(nuV[c], gUx[c], gUy[c], gUz[c], tX[c], tY[c], tZ[c]);
            },
            "buildTauRows_internal"
        );
    }

    // -------------------------
    // Boundary faces
    // -------------------------
    {
        const auto [nuB, gUxB, gUyB, gUzB, tXB_val, tYB_val, tZB_val, tXB_ref, tYB_ref, tZB_ref] =
            views(
                nu.boundaryData().value(),
                gradUx.boundaryData().value(),
                gradUy.boundaryData().value(),
                gradUz.boundaryData().value(),
                tauX.boundaryData().value(),
                tauY.boundaryData().value(),
                tauZ.boundaryData().value(),
                tauX.boundaryData().refValue(),
                tauY.boundaryData().refValue(),
                tauZ.boundaryData().refValue()
            );

        const auto nBF = mesh.boundaryMesh().offset().back();

        parallelFor(
            exec,
            {0, nBF},
            NEON_LAMBDA(const localIdx i) {
                Vec3 tx, ty, tz;
                buildTauRows(nuB[i], gUxB[i], gUyB[i], gUzB[i], tx, ty, tz);

                tXB_val[i] = tx;
                tXB_ref[i] = tx;
                tYB_val[i] = ty;
                tYB_ref[i] = ty;
                tZB_val[i] = tz;
                tZB_ref[i] = tz;
            },
            "buildTauRows_boundary"
        );
    }

    // -------------------------
    // Divergence
    // -------------------------
    computeDivVectorExp(tauX, surfInterpVec, divTauX, operatorScaling);
    computeDivVectorExp(tauY, surfInterpVec, divTauY, operatorScaling);
    computeDivVectorExp(tauZ, surfInterpVec, divTauZ, operatorScaling);

    // -------------------------
    // Assemble rhs
    // -------------------------
    {
        const auto [r, dx, dy, dz] = views(rhs, divTauX, divTauY, divTauZ);

        parallelFor(
            exec,
            {0, mesh.nCells()},
            NEON_LAMBDA(const localIdx c) {
                r[c][0] = dx[c];
                r[c][1] = dy[c];
                r[c][2] = dz[c];
            },
            "assembleDivTau"
        );
    }
}

KOKKOS_INLINE_FUNCTION
void atomicAddVec3(NeoN::Vec3* target, const NeoN::Vec3& v)
{
    Kokkos::atomic_add(&(*target)[0], v[0]);
    Kokkos::atomic_add(&(*target)[1], v[1]);
    Kokkos::atomic_add(&(*target)[2], v[2]);
}

KOKKOS_INLINE_FUNCTION
void atomicSubVec3(NeoN::Vec3* target, const NeoN::Vec3& v)
{
    Kokkos::atomic_sub(&(*target)[0], v[0]);
    Kokkos::atomic_sub(&(*target)[1], v[1]);
    Kokkos::atomic_sub(&(*target)[2], v[2]);
}

void computeLaplacianScalarGammaVectorExp(
    const FaceNormalGradient<Vec3>& faceNormalGradient, // snGrad scheme
    const SurfaceField<scalar>& gammaF,                 // e.g. nut on faces (all faces!)
    const VolumeField<Vec3>& U,                         // volVectorField
    Vector<Vec3>& lapU,                                 // cell vector result
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = U.mesh();
    const auto exec = U.exec();

    // caller decides whether lapU is accumulated or overwritten; here: overwrite
    fill(lapU, zero<Vec3>());

    // snGrad(U) on faces (Vec3 per face): component-wise snGrad of U
    // IMPORTANT: correctness on non-orth meshes depends on FaceNormalGradient implementation
    SurfaceField<Vec3> snGradU = faceNormalGradient.faceNormalGrad(U);

    const localIdx nIF = mesh.nInternalFaces();
    const localIdx nBF = mesh.nBoundaryFaces();
    const localIdx nFaces = nIF + nBF;

    const auto [owner, neighbour, faceCells] =
        views(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());

    const auto [SfAll, magSfAll, vol] =
        views(mesh.faceAreas(), mesh.magFaceAreas(), mesh.cellVolumes());

    const auto [gammaAll, snGradAll, res] = views(
        gammaF.internalVector(),  // must be size nIF+nBF
        snGradU.internalVector(), // must be size nIF+nBF
        lapU
    );

    NF_DEBUG_ASSERT(gammaAll.size() == nFaces, "gammaF must be defined on all faces (nIF+nBF)");
    NF_DEBUG_ASSERT(snGradAll.size() == nFaces, "snGradU must be defined on all faces (nIF+nBF)");

    // -------------------------
    // Internal faces: add/sub
    // -------------------------
    parallelFor(
        exec,
        {0, nIF},
        NEON_LAMBDA(const localIdx f) {
            const localIdx o = owner[f];
            const localIdx n = neighbour[f];

            // OpenFOAM:
            // SfGammaSn = gamma * magSf
            // flux = SfGammaSn * snGrad(U)
            const Vec3 flux = snGradAll[f] * (gammaAll[f] * magSfAll[f]);

            atomicAddVec3(&res[o], flux);
            atomicSubVec3(&res[n], flux);
        },
        "laplacianScalarGammaVec_Internal"
    );

    // -------------------------
    // Boundary faces: owner only
    // -------------------------
    parallelFor(
        exec,
        {nIF, nFaces},
        NEON_LAMBDA(const localIdx f) {
            const localIdx own = faceCells[f - nIF];

            const Vec3 flux = snGradAll[f] * (gammaAll[f] * magSfAll[f]);

            atomicAddVec3(&res[own], flux);
        },
        "laplacianScalarGammaVec_Boundary"
    );

    // -------------------------
    // Normalize by volume & scaling
    // -------------------------
    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { res[c] *= operatorScaling[c] / vol[c]; },
        "laplacianScalarGammaVec_Normalize"
    );
}


} // namespace NeoN::finiteVolume::cellCentred
