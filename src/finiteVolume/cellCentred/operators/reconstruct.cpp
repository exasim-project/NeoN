// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/reconstruct.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/vector/vector.hpp"

namespace NeoN::finiteVolume::cellCentred
{

VolumeField<Vec3> reconstruct(const SurfaceField<scalar>& ssf)
{
    const auto& mesh = ssf.mesh();
    const auto exec = ssf.exec();
    const auto nCells = mesh.nCells();
    const auto nInt = mesh.nInternalFaces();
    const auto nBnd = mesh.nBoundaryFaces();
    const auto nProc = mesh.nProcBoundaryFaces();

    const auto sf = mesh.faceNormals().view();  // Vec3 area vectors
    const auto magSf = mesh.faceAreas().view(); // |Sf|
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto ssfI = ssf.internalVector().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    const auto bSf = mesh.boundaryMesh().faceNormals().view();
    const auto bMag = mesh.boundaryMesh().faceAreas().view();
    const auto ssfB = ssf.boundaryData().value().view();

    // Per-cell symmetric 3x3 accumulators (6 comps) + rhs Vec3.
    NeoN::Vector<NeoN::scalar> gxxVec(exec, nCells, 0.0), gxyVec(exec, nCells, 0.0),
        gxzVec(exec, nCells, 0.0), gyyVec(exec, nCells, 0.0), gyzVec(exec, nCells, 0.0),
        gzzVec(exec, nCells, 0.0);
    NeoN::Vector<NeoN::Vec3> b(exec, nCells, NeoN::Vec3 {0.0, 0.0, 0.0});
    auto gxx = gxxVec.view(), gxy = gxyVec.view(), gxz = gxzVec.view(), gyy = gyyVec.view(),
         gyz = gyzVec.view(), gzz = gzzVec.view();
    auto bv = b.view();

    auto scatter =
        NEON_LAMBDA(const NeoN::localIdx c, const NeoN::Vec3& sf, NeoN::scalar mg, NeoN::scalar val)
    {
        const NeoN::scalar im = 1.0 / mg;
        Kokkos::atomic_add(&gxx[c], sf[0] * sf[0] * im);
        Kokkos::atomic_add(&gxy[c], sf[0] * sf[1] * im);
        Kokkos::atomic_add(&gxz[c], sf[0] * sf[2] * im);
        Kokkos::atomic_add(&gyy[c], sf[1] * sf[1] * im);
        Kokkos::atomic_add(&gyz[c], sf[1] * sf[2] * im);
        Kokkos::atomic_add(&gzz[c], sf[2] * sf[2] * im);
        Kokkos::atomic_add(&bv[c][0], sf[0] * im * val);
        Kokkos::atomic_add(&bv[c][1], sf[1] * im * val);
        Kokkos::atomic_add(&bv[c][2], sf[2] * im * val);
    };

    NeoN::parallelFor(
        exec,
        {0, nInt},
        NEON_LAMBDA(const NeoN::localIdx f) {
            scatter(own[f], sf[f], magSf[f], ssfI[f]); // surfaceSum: +owner
            scatter(nei[f], sf[f], magSf[f], ssfI[f]); // surfaceSum: +neighbour
        },
        "reconstruct::scatterInternal"
    );
    // Physical boundary faces: bSf points out of the owner cell and the stored face value is
    // already the outward flux, so no sign correction is needed.
    NeoN::parallelFor(
        exec,
        {0, nBnd},
        NEON_LAMBDA(const NeoN::localIdx bf) { scatter(bOwn[bf], bSf[bf], bMag[bf], ssfB[bf]); },
        "reconstruct::scatterBoundary"
    );
    if (nProc > 0)
    {
        // Processor (coupled) faces occupy the trailing boundary slots. Skipping them would
        // make the reconstruction decomposition-dependent for cells on a partition interface.
        // bSf points out of the LOCAL cell, but the stored flux keeps the global owner->neighbour
        // sense, so flip it where the local cell is the neighbour of the cross-rank face.
        // boundaryMesh().weights() carries that sign (same convention as BoundedDiv).
        const auto isOwnerV = mesh.boundaryMesh().weights().view();
        NeoN::parallelFor(
            exec,
            {0, nProc},
            NEON_LAMBDA(const NeoN::localIdx procFacei) {
                const auto bf = nBnd + procFacei;
                const auto val = (isOwnerV[bf] > 0.0) ? ssfB[bf] : -ssfB[bf];
                scatter(bOwn[bf], bSf[bf], bMag[bf], val);
            },
            "reconstruct::scatterProcBoundary"
        );
    }

    // 'extrapolated' on physical patches + 'processor' on coupled patches: the trailing
    // correctBoundaryConditions() then fills physical boundary values from the owner cell and
    // exchanges the neighbour value across rank boundaries. A plain 'calculated' BC is a no-op
    // and would leave the whole boundary field at zero.
    VolumeField<Vec3> res(
        exec,
        "reconstruct(" + ssf.name + ")",
        mesh,
        createExtrapolatedBCs<VolumeBoundary<NeoN::Vec3>>(mesh)
    );
    auto r = res.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const NeoN::localIdx c) {
            // invert symmetric [[a,d,e],[d,ff,g],[e,g,h]] & b
            const NeoN::scalar a0 = gxx[c], f0 = gyy[c], h0 = gzz[c];
            const NeoN::scalar d = gxy[c], e = gxz[c], g = gyz[c];
            const NeoN::scalar tr = a0 + f0 + h0;
            const NeoN::Vec3 rhs = bv[c];

            // Regularise empty (zero-area) directions so the in-plane block still inverts on 2D
            // meshes. A planar mesh contributes no face area along its empty direction, leaving a
            // zero diagonal there (e.g. gzz==0 on a z-normal mesh) and hence a singular tensor.
            // Setting that decoupled diagonal to the trace makes det!=0 while leaving the in-plane
            // 2x2 result (and the ~0 empty-direction result) unchanged. Only coordinate-aligned
            // null components are stripped this way; an oblique one falls through to the ridge
            // below.
            const NeoN::scalar reg = 1e-9 * tr;
            NeoN::scalar a = (a0 < reg) ? tr : a0;
            NeoN::scalar ff = (f0 < reg) ? tr : f0;
            NeoN::scalar h = (h0 < reg) ? tr : h0;

            NeoN::scalar det = a * (ff * h - g * g) + d * (e * g - d * h) + e * (d * g - e * ff);

            // The axis-aligned strip above cannot see a null direction that is oblique to the
            // coordinate axes (a rank-deficient mesh in a rotated plane keeps all three diagonals
            // above reg while the tensor stays singular). Fall back to an isotropic ridge in that
            // case: it is rotation invariant and perturbs the well-resolved directions by only
            // O(1e-8) relative, while the (numerically ~zero) null component stays negligible.
            // Without it det==0 and the whole cell would be forced to zero.
            if (!(Kokkos::fabs(det) > 1e-12 * tr * tr * tr))
            {
                const NeoN::scalar ridge = 1e-8 * tr;
                a = a0 + ridge;
                ff = f0 + ridge;
                h = h0 + ridge;
                det = a * (ff * h - g * g) + d * (e * g - d * h) + e * (d * g - e * ff);
            }

            const NeoN::scalar c00 = ff * h - g * g;
            const NeoN::scalar c01 = e * g - d * h;
            const NeoN::scalar c02 = d * g - e * ff;
            const NeoN::scalar id = (Kokkos::fabs(det) > 1e-300) ? 1.0 / det : 0.0;
            const NeoN::scalar c11 = a * h - e * e;
            const NeoN::scalar c12 = e * d - a * g;
            const NeoN::scalar c22 = a * ff - d * d;
            r[c][0] = id * (c00 * rhs[0] + c01 * rhs[1] + c02 * rhs[2]);
            r[c][1] = id * (c01 * rhs[0] + c11 * rhs[1] + c12 * rhs[2]);
            r[c][2] = id * (c02 * rhs[0] + c12 * rhs[1] + c22 * rhs[2]);
        },
        "reconstruct::solve"
    );
    res.correctBoundaryConditions();
    return res;
}

} // namespace NeoN::finiteVolume::cellCentred
