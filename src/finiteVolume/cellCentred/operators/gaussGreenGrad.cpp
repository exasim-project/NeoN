// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/tensor.hpp"

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

    const auto [boundaryFaceOwners, faceOwners, faceNeighbors, faceAreaS, surfV] = views(
        mesh.boundaryMesh().faceOwners(),
        mesh.faceOwners(),
        mesh.faceNeighbors(),
        mesh.faceNormals(),
        mesh.cellVolumes()
    );
    const auto surfPhif = phif.internalVector().view();
    const auto surfPhifB = phif.boundaryData().value().view();

    auto nInternalFaces = mesh.nInternalFaces();
    auto nBoundaryFaces = mesh.nBoundaryFaces();

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
            Kokkos::atomic_add(&surfGradPhi[faceOwners[i]], flux);    // +S_f * φ_f
            Kokkos::atomic_sub(&surfGradPhi[faceNeighbors[i]], flux); // −S_f * φ_f
        },
        "computeGradInternal"
    );

    // Boundary faces: only the owner cell is on this rank.
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = boundaryFaceOwners[bfi];
            Vec3 valueOwn = faceAreaS[nInternalFaces + bfi] * surfPhifB[bfi];
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
    const VolumeField<scalar>& phi,
    VolumeField<Vec3>& gradPhi,
    [[maybe_unused]] const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = gradPhi.exec();
    const auto boundaryConditions = phi.boundaryConditions();

    auto gradInternal = gradPhi.internalVector().view();
    auto gradBoundary = gradPhi.boundaryData().value().view();
    const auto
        [phiInternal, phiBoundaryValue, phiBoundaryRefGrad, faceCells, deltaCoeffs, normals] =
            views(
                phi.internalVector(),
                phi.boundaryData().value(),
                phi.boundaryData().refGrad(),
                mesh.boundaryMesh().faceOwners(),
                mesh.boundaryMesh().deltaCoeffs(),
                mesh.boundaryMesh().faceUnitNormals()
            );

    for (localIdx patchID = 0; patchID < mesh.nBoundaries(); ++patchID)
    {
        const auto attrs = boundaryConditions[static_cast<size_t>(patchID)].attributes();
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

                    // Extrapolate internal gradient
                    Vec3 g = gradInternal[owner];

                    // Compute snGrad
                    const scalar snGrad =
                        (phiBoundaryValue[i] - phiInternal[owner]) * deltaCoeffs[i];
                    const Vec3 n = normals[i];

                    // Normal reconstruction
                    const scalar nDotG = n[0] * g[0] + n[1] * g[1] + n[2] * g[2];

                    g += n * (snGrad - nDotG);

                    gradBoundary[i] = g;
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

                    // Extrapolate internal gradient
                    Vec3 g = gradInternal[owner];

                    const Vec3 n = normals[i];

                    // snGrad from BC
                    const scalar snGrad = phiBoundaryRefGrad[i];

                    // 2) Normal reconstruction
                    const scalar nDotG = n[0] * g[0] + n[1] * g[1] + n[2] * g[2];

                    g += n * (snGrad - nDotG);

                    gradBoundary[i] = g;
                },
                "computeGradBoundaryRefGrad"
            );
        }
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

void GaussGreenGrad::grad(
    const VolumeField<scalar>& phi, VolumeField<Vec3>& gradPhi, const dsl::Coeff operatorScaling
) const
{
    fill(gradPhi.internalVector(), zero<Vec3>());

    computeGrad(phi, surfaceInterpolation_, gradPhi.internalVector(), operatorScaling);
    computeBoundaryGrad(phi, gradPhi, operatorScaling);
}

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

// ---- Tensor gradient implementation ----------------------------------------

KOKKOS_INLINE_FUNCTION
void atomicAddTensor(Tensor* target, size_t row, size_t col, scalar value)
{
    Kokkos::atomic_add(&(*target)(row, col), value);
}

KOKKOS_INLINE_FUNCTION
void atomicSubTensor(Tensor* target, size_t row, size_t col, scalar value)
{
    Kokkos::atomic_sub(&(*target)(row, col), value);
}

void computeGradTensor(
    const VolumeField<Vec3>& u,
    const SurfaceInterpolation<Vec3>& surfInterpVec,
    Vector<Tensor>& gradU,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = u.mesh();
    const auto exec = gradU.exec();

    SurfaceField<Vec3> uf(exec, "Uf", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh));
    surfInterpVec.interpolate(u, uf);

    auto gT = gradU.view();

    const auto [owner, nei, SfAll, V, bFaceCells] = views(
        mesh.faceOwners(),
        mesh.faceNeighbors(),
        mesh.faceNormals(),
        mesh.cellVolumes(),
        mesh.boundaryMesh().faceOwners()
    );
    const auto ufInt = uf.internalVector().view();
    const auto ufBound = uf.boundaryData().value().view();

    const localIdx nInt = mesh.nInternalFaces();
    const localIdx nBnd = mesh.nBoundaryFaces();

    parallelFor(
        exec,
        {0, nInt},
        NEON_LAMBDA(const localIdx f) {
            const Vec3 sf = SfAll[f];
            const Vec3 faceU = ufInt[f];
            const auto o = owner[f];
            const auto n = nei[f];
            // gradU(row,col) += Sf[col] * U[row]  (Gauss-Green)
            for (size_t row = 0; row < 3; ++row)
            {
                for (size_t col = 0; col < 3; ++col)
                {
                    const scalar c = sf[col] * faceU[row];
                    atomicAddTensor(&gT[o], row, col, c);
                    atomicSubTensor(&gT[n], row, col, c);
                }
            }
        },
        "computeGradTensorInternal"
    );

    parallelFor(
        exec,
        {0, nBnd},
        NEON_LAMBDA(const localIdx bi) {
            const auto o = bFaceCells[bi];
            // TODO Issue #515
            const Vec3 sf = SfAll[nInt + bi];
            const Vec3 faceU = ufBound[bi];
            for (size_t row = 0; row < 3; ++row)
            {
                for (size_t col = 0; col < 3; ++col)
                {
                    atomicAddTensor(&gT[o], row, col, sf[col] * faceU[row]);
                }
            }
        },
        "computeGradTensorBoundary"
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) {
            const scalar s = operatorScaling[c] / V[c];
            gT[c] *= s;
        },
        "computeGradTensorCells"
    );
}

void computeBoundaryGradTensor(const VolumeField<Vec3>& u, VolumeField<Tensor>& gradU)
{
    const auto& mesh = u.mesh();
    const auto exec = u.exec();
    const auto& offsets = mesh.boundaryMesh().offset();

    const auto bcs = u.boundaryConditions();

    auto gTInt = gradU.internalVector().view();
    auto gTB = gradU.boundaryData().value().view();

    const auto [UInt, UB, URefGradB, faceCells, deltaCoeffs, nHat] = views(
        u.internalVector(),
        u.boundaryData().value(),
        u.boundaryData().refGrad(),
        mesh.boundaryMesh().faceOwners(),
        mesh.boundaryMesh().deltaCoeffs(),
        mesh.boundaryMesh().faceUnitNormals()
    );

    for (localIdx patchID = 0; patchID < static_cast<localIdx>(offsets.size() - 1); ++patchID)
    {
        const localIdx start = offsets[static_cast<size_t>(patchID)];
        const localIdx end = offsets[static_cast<size_t>(patchID + 1)];
        if (start == end) continue;

        const auto attrs = bcs[static_cast<size_t>(patchID)].attributes();

        parallelFor(
            exec,
            {static_cast<size_t>(start), static_cast<size_t>(end)},
            NEON_LAMBDA(const localIdx i) {
                const auto owner = faceCells[i];
                const Vec3 n = nHat[i];
                Tensor g = gTInt[owner];

                Vec3 snGrad;
                if (attrs.fixesValue) snGrad = (UB[i] - UInt[owner]) * deltaCoeffs[i];
                else
                    snGrad = URefGradB[i];

                // Reconstruct each row: g(row,:) += n * (snGrad[row] - n · g(row,:))
                for (size_t row = 0; row < 3; ++row)
                {
                    const Vec3 gRow(g(row, 0), g(row, 1), g(row, 2));
                    const scalar nDotG = n[0] * gRow[0] + n[1] * gRow[1] + n[2] * gRow[2];
                    const Vec3 corrected = gRow + n * (snGrad[row] - nDotG);
                    g(row, 0) = corrected[0];
                    g(row, 1) = corrected[1];
                    g(row, 2) = corrected[2];
                }
                gTB[i] = g;
            },
            "computeGradTensorBoundaryReconstruct"
        );
    }
}

void GaussGreenGrad::gradTensor(
    const VolumeField<Vec3>& u, VolumeField<Tensor>& gradU, const dsl::Coeff operatorScaling
) const
{
    fill(gradU.internalVector(), zero<Tensor>());
    computeGradTensor(u, surfaceInterpolationVec_, gradU.internalVector(), operatorScaling);
    computeBoundaryGradTensor(u, gradU);
}

VolumeField<Tensor>
GaussGreenGrad::gradTensor(const VolumeField<Vec3>& u, const dsl::Coeff operatorScaling) const
{
    auto calcBC = createCalculatedBCs<VolumeBoundary<Tensor>>(u.mesh());
    VolumeField<Tensor> gradU(u.exec(), "gradU", u.mesh(), calcBC);
    fill(gradU.internalVector(), zero<Tensor>());

    computeGradTensor(u, surfaceInterpolationVec_, gradU.internalVector(), operatorScaling);
    computeBoundaryGradTensor(u, gradU);
    return gradU;
}

} // namespace NeoN
