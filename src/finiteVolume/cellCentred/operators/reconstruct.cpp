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

    const auto Sf = mesh.faceNormals().view();   // Vec3 area vectors
    const auto magSf = mesh.faceAreas().view();  // |Sf|
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto ssfI = ssf.internalVector().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    const auto bSf = mesh.boundaryMesh().faceNormals().view();
    const auto bMag = mesh.boundaryMesh().faceAreas().view();
    const auto ssfB = ssf.boundaryData().value().view();

    // Per-cell symmetric 3x3 accumulators (6 comps) + rhs Vec3.
    NeoN::Vector<NeoN::scalar> Gxx(exec, nCells, 0.0), Gxy(exec, nCells, 0.0),
        Gxz(exec, nCells, 0.0), Gyy(exec, nCells, 0.0), Gyz(exec, nCells, 0.0),
        Gzz(exec, nCells, 0.0);
    NeoN::Vector<NeoN::Vec3> b(exec, nCells, NeoN::Vec3 {0.0, 0.0, 0.0});
    auto gxx = Gxx.view(), gxy = Gxy.view(), gxz = Gxz.view(), gyy = Gyy.view(),
         gyz = Gyz.view(), gzz = Gzz.view();
    auto bv = b.view();

    auto scatter = KOKKOS_LAMBDA(
        const NeoN::localIdx c, const NeoN::Vec3& sf, NeoN::scalar mg, NeoN::scalar val
    )
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
        exec, {0, nInt},
        KOKKOS_LAMBDA(const NeoN::localIdx f) {
            scatter(own[f], Sf[f], magSf[f], ssfI[f]); // surfaceSum: +owner
            scatter(nei[f], Sf[f], magSf[f], ssfI[f]); // surfaceSum: +neighbour
        },
        "reconstruct::scatterInternal"
    );
    NeoN::parallelFor(
        exec, {0, nBnd},
        KOKKOS_LAMBDA(const NeoN::localIdx bf) {
            scatter(bOwn[bf], bSf[bf], bMag[bf], ssfB[bf]);
        },
        "reconstruct::scatterBoundary"
    );

    VolumeField<Vec3> res(
        exec, "reconstruct(" + ssf.name + ")", mesh,
        createCalculatedBCs<VolumeBoundary<NeoN::Vec3>>(mesh)
    );
    auto r = res.internalVector().view();
    NeoN::parallelFor(
        exec, {0, nCells},
        KOKKOS_LAMBDA(const NeoN::localIdx c) {
            // invert symmetric [[a,d,e],[d,ff,g],[e,g,h]] & b
            NeoN::scalar a = gxx[c], ff = gyy[c], h = gzz[c];
            const NeoN::scalar d = gxy[c], e = gxz[c], g = gyz[c];
            // Regularise empty (zero-area) directions so the in-plane block still
            // inverts on 2D meshes. OpenFOAM keeps the empty front/back faces in the
            // surfaceSum, giving a full-rank tensor; NeoN drops them, leaving a zero
            // diagonal in the empty direction (e.g. gzz==0 on a planar z-normal mesh).
            // Setting that decoupled diagonal to the trace makes det!=0 while leaving
            // the in-plane 2x2 result (and the ~0 empty-direction result) unchanged —
            // matching fvc::reconstruct componentwise.
            const NeoN::scalar tr = a + ff + h;
            const NeoN::scalar reg = 1e-9 * tr;
            if (a < reg) a = tr;
            if (ff < reg) ff = tr;
            if (h < reg) h = tr;
            const NeoN::scalar c00 = ff * h - g * g;
            const NeoN::scalar c01 = e * g - d * h;
            const NeoN::scalar c02 = d * g - e * ff;
            const NeoN::scalar det = a * c00 + d * c01 + e * c02;
            const NeoN::scalar id = (Kokkos::fabs(det) > 1e-300) ? 1.0 / det : 0.0;
            const NeoN::scalar c11 = a * h - e * e;
            const NeoN::scalar c12 = e * d - a * g;
            const NeoN::scalar c22 = a * ff - d * d;
            const NeoN::Vec3 rhs = bv[c];
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
