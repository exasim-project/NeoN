// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

KOKKOS_INLINE_FUNCTION
void atomicAdd(Vec3* target, const Vec3& value)
{
    Kokkos::atomic_add(&(*target)[0], value[0]);
    Kokkos::atomic_add(&(*target)[1], value[1]);
    Kokkos::atomic_add(&(*target)[2], value[2]);
}

KOKKOS_INLINE_FUNCTION
void atomicSub(Vec3* target, const Vec3& value)
{
    Kokkos::atomic_sub(&(*target)[0], value[0]);
    Kokkos::atomic_sub(&(*target)[1], value[1]);
    Kokkos::atomic_sub(&(*target)[2], value[2]);
}

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

    // TODO use NeoN::atomic_
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            Vec3 flux = faceAreaS[i] * surfPhif[i];
            Kokkos::atomic_add(&surfGradPhi[surfOwner[i]], flux);
            Kokkos::atomic_sub(&surfGradPhi[surfNeighbour[i]], flux);
        },
        "computeGradInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, surfPhif.size()},
        NEON_LAMBDA(const localIdx i) {
            auto own = surfFaceCells[i - nInternalFaces];
            Vec3 valueOwn = faceAreaS[i] * surfPhif[i];
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

void computeBoundaryGrad(
    const VolumeField<scalar>& phi, VolumeField<Vec3>& gradPhi, const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = gradPhi.exec();
    const auto boundaryConditions = phi.boundaryConditions();

    // const auto& offsets = mesh.boundaryMesh().offset();
    auto gradInternal = gradPhi.internalVector().view();
    auto gradBoundary = gradPhi.boundaryData().value().view();
    const auto
        [phiInternal, phiBoundaryValue, phiBoundaryRefGrad, faceCells, deltaCoeffs, normals] =
            views(
                phi.internalVector(),
                phi.boundaryData().value(),
                phi.boundaryData().refGrad(),
                mesh.boundaryMesh().faceCells(),
                mesh.boundaryMesh().deltaCoeffs(),
                mesh.boundaryMesh().nf()
            );

    // for (localIdx patchID = 0; patchID < static_cast<localIdx>(offsets.size() - 1); ++patchID)
    for (localIdx patchID = 0; patchID < mesh.nBoundaries(); ++patchID)
    {
        const auto attrs = boundaryConditions[patchID].attributes();
        // const localIdx start = offsets[patchID];
        // const localIdx end   = offsets[patchID + 1];
        const auto [start, end] = phi.boundaryData().range(patchID);

        if (start == end)
        {
            continue;
        }

        if (attrs.fixesValue)
        {
            parallelFor(
                exec,
                {start, end},
                NEON_LAMBDA(const localIdx i) {
                    const auto owner = faceCells[i];

                    // 1) Extrapolate internal gradient
                    Vec3 g = gradInternal[owner];

                    // 2) Compute snGrad
                    const scalar snGrad =
                        (phiBoundaryValue[i] - phiInternal[owner]) * deltaCoeffs[i];
                    const Vec3 n = normals[i];

                    // 3) Normal reconstruction
                    const scalar nDotG = n[0] * g[0] + n[1] * g[1] + n[2] * g[2];

                    g += n * (snGrad - nDotG);

                    gradBoundary[i] = Vec3 {1, 2, 3};
                },
                "computeGradBoundaryFixedValue"
            );
        }
        else
        {
            parallelFor(
                exec,
                {start, end},
                NEON_LAMBDA(const localIdx i) {
                    const auto owner = faceCells[i];

                    // 1) Extrapolate internal gradient
                    Vec3 g = gradInternal[owner];

                    const Vec3 n = normals[i];

                    // snGrad from BC
                    const scalar snGrad = phiBoundaryRefGrad[i];

                    // 2) Normal reconstruction
                    const scalar nDotG = n[0] * g[0] + n[1] * g[1] + n[2] * g[2];

                    g += n * (snGrad - nDotG);

                    gradBoundary[i] = Vec3 {5, 6, 7};
                },
                "computeGradBoundaryRefGrad"
            );
        }
    }
}

void computeGradVec(
    const VolumeField<Vec3>& U,
    const SurfaceInterpolation<Vec3>& surfInterpVec,
    Vector<Vec3>& gradUx,
    Vector<Vec3>& gradUy,
    Vector<Vec3>& gradUz,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = U.mesh();
    const auto exec = gradUx.exec();

    SurfaceField<Vec3> Uf(exec, "Uf", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh));
    surfInterpVec.interpolate(U, Uf);

    auto gUx = gradUx.view();
    auto gUy = gradUy.view();
    auto gUz = gradUz.view();

    const auto [UfAll, owner, nei, SfAll, V, bFaceCells] = views(
        Uf.internalVector(), // expected: size >= nFaces
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.faceAreas(), // <-- IMPORTANT: ALL faces
        mesh.cellVolumes(),
        mesh.boundaryMesh().faceCells() // boundary faces only
    );

    const localIdx nInt = mesh.nInternalFaces();
    const localIdx nBnd = mesh.boundaryMesh().offset().back();
    const localIdx nFaces = nInt + nBnd;

    // (Optional) debug asserts
    // NF_ASSERT_EQUAL(static_cast<size_t>(nFaces), UfAll.size());

    parallelFor(
        exec,
        {0, nInt},
        NEON_LAMBDA(const localIdx f) {
            const Vec3 sf = SfAll[f];
            const Vec3 uf = UfAll[f];

            const Vec3 fluxX = sf * uf[0];
            const Vec3 fluxY = sf * uf[1];
            const Vec3 fluxZ = sf * uf[2];

            const auto o = owner[f];
            const auto n = nei[f];

            atomicAdd(&gUx[o], fluxX);
            atomicSub(&gUx[n], fluxX);

            atomicAdd(&gUy[o], fluxY);
            atomicSub(&gUy[n], fluxY);

            atomicAdd(&gUz[o], fluxZ);
            atomicSub(&gUz[n], fluxZ);
        },
        "computeGradVecInternal"
    );

    parallelFor(
        exec,
        {nInt, nFaces},
        NEON_LAMBDA(const localIdx f) {
            const localIdx bi = f - nInt; // boundary-face index
            const auto o = bFaceCells[bi];

            const Vec3 sf = SfAll[f];
            const Vec3 uf = UfAll[f];

            atomicAdd(&gUx[o], sf * uf[0]);
            atomicAdd(&gUy[o], sf * uf[1]);
            atomicAdd(&gUz[o], sf * uf[2]);
        },
        "computeGradVecBoundary"
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) {
            const scalar s = operatorScaling[c] / V[c];
            gUx[c] *= s;
            gUy[c] *= s;
            gUz[c] *= s;
        },
        "computeGradVecCells"
    );
}
/*
void computeGradVec(
    const VolumeField<Vec3>& U,
    const SurfaceInterpolation<Vec3>& surfInterpVec,
    Vector<Vec3>& gradUx,   // internal storage only
    Vector<Vec3>& gradUy,
    Vector<Vec3>& gradUz,
    const dsl::Coeff operatorScaling
)
{
    const auto& mesh = U.mesh();
    const auto exec = gradUx.exec();

    SurfaceField<Vec3> Uf(
        exec, "Uf", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
    );
    surfInterpVec.interpolate(U, Uf);

    auto gUx = gradUx.view();
    auto gUy = gradUy.view();
    auto gUz = gradUz.view();

    const auto [bFaceCells, Sf, UfInt, owner, nei, V] = views(
        mesh.boundaryMesh().faceCells(),
        mesh.boundaryMesh().sf(),
        Uf.internalVector(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.cellVolumes()
    );

    const localIdx nIntFaces = mesh.nInternalFaces();

    // internal faces
    parallelFor(
        exec,
        {0, nIntFaces},
        NEON_LAMBDA(const localIdx f)
        {
            const Vec3 sf = Sf[f];
            const Vec3 uf = UfInt[f];

            const Vec3 fluxX = sf * uf[0];
            const Vec3 fluxY = sf * uf[1];
            const Vec3 fluxZ = sf * uf[2];

            const auto o = owner[f];
            const auto n = nei[f];

            Kokkos::atomic_add(&gUx[o], fluxX);
            Kokkos::atomic_sub(&gUx[n], fluxX);

            Kokkos::atomic_add(&gUy[o], fluxY);
            Kokkos::atomic_sub(&gUy[n], fluxY);

            Kokkos::atomic_add(&gUz[o], fluxZ);
            Kokkos::atomic_sub(&gUz[n], fluxZ);
        },
        "computeGradVecInternal"
    );

    // boundary faces (faces stored after internal)
    parallelFor(
        exec,
        {nIntFaces, static_cast<localIdx>(UfInt.size())},
        NEON_LAMBDA(const localIdx f)
        {
            const localIdx bi = f - nIntFaces;
            const auto o = bFaceCells[bi];

            const Vec3 sf = Sf[f];
            const Vec3 uf = UfInt[f];

            Kokkos::atomic_add(&gUx[o], sf * uf[0]);
            Kokkos::atomic_add(&gUy[o], sf * uf[1]);
            Kokkos::atomic_add(&gUz[o], sf * uf[2]);
        },
        "computeGradVecBoundary"
    );

    // scale by 1/V
    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c)
        {
            const scalar s = operatorScaling[c] / V[c];
            gUx[c] *= s;
            gUy[c] *= s;
            gUz[c] *= s;
        },
        "computeGradVecCells"
    );
}
*/

void computeBoundaryGradVec(
    const VolumeField<Vec3>& U,
    VolumeField<Vec3>& gradUx,
    VolumeField<Vec3>& gradUy,
    VolumeField<Vec3>& gradUz
)
{
    const auto& mesh = U.mesh();
    const auto exec = U.exec();
    const auto& offsets = mesh.boundaryMesh().offset();

    const auto bcs = U.boundaryConditions();

    auto gUxInt = gradUx.internalVector().view();
    auto gUyInt = gradUy.internalVector().view();
    auto gUzInt = gradUz.internalVector().view();

    auto gUxB = gradUx.boundaryData().value().view();
    auto gUyB = gradUy.boundaryData().value().view();
    auto gUzB = gradUz.boundaryData().value().view();

    const auto [UInt, UB, URefGradB, faceCells, deltaCoeffs, nHat] = views(
        U.internalVector(),
        U.boundaryData().value(),
        U.boundaryData().refGrad(),
        mesh.boundaryMesh().faceCells(),
        mesh.boundaryMesh().deltaCoeffs(),
        mesh.boundaryMesh().nf()
    );

    for (localIdx patchID = 0; patchID < static_cast<localIdx>(offsets.size() - 1); ++patchID)
    {
        const localIdx start = offsets[patchID];
        const localIdx end = offsets[patchID + 1];
        if (start == end) continue;

        const auto attrs = bcs[patchID].attributes();

        parallelFor(
            exec,
            {static_cast<size_t>(start), static_cast<size_t>(end)},
            NEON_LAMBDA(const localIdx i) {
                const auto owner = faceCells[i];
                const Vec3 n = nHat[i];

                Vec3 gx = gUxInt[owner];
                Vec3 gy = gUyInt[owner];
                Vec3 gz = gUzInt[owner];

                Vec3 snGrad;
                if (attrs.fixesValue)
                {
                    const Vec3 dU = (UB[i] - UInt[owner]) * deltaCoeffs[i];
                    snGrad = dU; // component-wise snGrad
                }
                else
                {
                    snGrad = URefGradB[i]; // component-wise prescribed snGrad
                }

                // normal reconstruction per component row:
                // gx += n*(snGrad_x - n·gx), etc.
                const scalar ndgx = n[0] * gx[0] + n[1] * gx[1] + n[2] * gx[2];
                const scalar ndgy = n[0] * gy[0] + n[1] * gy[1] + n[2] * gy[2];
                const scalar ndgz = n[0] * gz[0] + n[1] * gz[1] + n[2] * gz[2];

                gx += n * (snGrad[0] - ndgx);
                gy += n * (snGrad[1] - ndgy);
                gz += n * (snGrad[2] - ndgz);

                gUxB[i] = gx;
                gUyB[i] = gy;
                gUzB[i] = gz;
            },
            "computeGradVecBoundaryReconstruct"
        );
    }
}


GaussGreenGrad::GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh)
    : Base(exec, mesh),
      surfaceInterpolation_(exec, mesh, std::make_unique<Linear<scalar>>(exec, mesh, Dictionary())),
      surfaceInterpolationVec_(
          exec, mesh, std::make_unique<Linear<Vec3>>(exec, mesh, Dictionary())
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
    computeBoundaryGrad(phi, gradPhi, operatorScaling);
    return gradPhi;
}

GradVecField
GaussGreenGrad::grad(const VolumeField<Vec3>& U, const dsl::Coeff operatorScaling) const
{
    auto calcBC = createCalculatedBCs<VolumeBoundary<Vec3>>(U.mesh());

    GradVecField G {
        VolumeField<Vec3>(U.exec(), "gradUx", U.mesh(), calcBC),
        VolumeField<Vec3>(U.exec(), "gradUy", U.mesh(), calcBC),
        VolumeField<Vec3>(U.exec(), "gradUz", U.mesh(), calcBC)
    };

    fill(G.gradUx.internalVector(), zero<Vec3>());
    fill(G.gradUy.internalVector(), zero<Vec3>());
    fill(G.gradUz.internalVector(), zero<Vec3>());

    computeGradVec(
        U,
        surfaceInterpolationVec_,
        G.gradUx.internalVector(),
        G.gradUy.internalVector(),
        G.gradUz.internalVector(),
        operatorScaling
    );

    computeBoundaryGradVec(U, G.gradUx, G.gradUy, G.gradUz);

    return G;
}

} // namespace NeoN
