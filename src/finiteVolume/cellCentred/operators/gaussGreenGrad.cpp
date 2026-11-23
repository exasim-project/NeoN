// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/cellToFaceStencil.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/helpers/exceptions.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief returns the internal cell-to-face stencil of the mesh, built once and cached in
** mesh.stencilDB()
*/
std::shared_ptr<const SegmentedVector<localIdx, localIdx>>
readOrCreateCellFaces(const UnstructuredMesh& mesh)
{
    using CellFaces = SegmentedVector<localIdx, localIdx>;
    auto& db = mesh.stencilDB();
    if (!db.contains("GaussGreenGradCellFaces"))
    {
        db.insert(
            std::string("GaussGreenGradCellFaces"),
            std::shared_ptr<const CellFaces>(
                std::make_shared<CellFaces>(CellToFaceStencil(mesh).computeInternalStencil())
            )
        );
    }
    return db.get<std::shared_ptr<const CellFaces>>("GaussGreenGradCellFaces");
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

    const auto [boundaryFaceOwners, faceOwners, faceNormals, surfV] = views(
        mesh.boundaryMesh().faceOwners(), mesh.faceOwners(), mesh.faceNormals(), mesh.cellVolumes()
    );
    const auto surfPhif = phif.internalVector().view();
    const auto surfPhifB = phif.boundaryData().value().view();

    auto nBoundaryFaces = mesh.nBoundaryFaces();

    // Green-Gauss gradient theorem: ∇φ_C = (1/V_C) * sum_f S_f * φ_f
    //
    // S_f points from owner to neighbour by construction (valid for all internal faces).
    //   owner cell:     S_f is the outward area vector  →  +S_f * φ_f  (add)
    //   neighbour cell: S_f points inward to neighbour  → −S_f * φ_f  (subtract)
    // Each cell sums its own internal faces, so no atomics are needed.
    const auto cellFaces = readOrCreateCellFaces(mesh);
    const auto [cellFacesValues, cellFacesSegments] =
        views(cellFaces->values(), cellFaces->segments());
    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx celli) {
            Vec3 grad = zero<Vec3>();
            for (auto i = cellFacesSegments[celli]; i < cellFacesSegments[celli + 1]; ++i)
            {
                const auto fi = cellFacesValues[i];
                const Vec3 flux = faceNormals[fi] * surfPhif[fi];
                if (faceOwners[fi] == celli)
                {
                    grad += flux; // +S_f * φ_f
                }
                else
                {
                    grad -= flux; // −S_f * φ_f
                }
            }
            surfGradPhi[celli] += grad;
        },
        "computeGradInternal"
    );

    // Boundary faces: only the owner cell is on this rank.
    const auto bFaceNormals = mesh.boundaryMesh().faceNormals().view();
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = boundaryFaceOwners[bfi];
            Vec3 valueOwn = bFaceNormals[bfi] * surfPhifB[bfi];
            Kokkos::atomic_add(&surfGradPhi[own], valueOwn);
        },
        "computeGradBoundary"
    );

    auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        const auto bSf = mesh.boundaryMesh().faceNormals().view();
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                auto bcfacei = nBoundaryFaces + procFacei;
                auto own = boundaryFaceOwners[bcfacei];
                Vec3 flux = bSf[bcfacei] * surfPhifB[bcfacei];
                Kokkos::atomic_add(&surfGradPhi[own], flux);
            },
            "computeProcGradBoundary"
        );
    }

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

    auto gradInternal = gradPhi.internalVector().view();
    auto gradBoundary = gradPhi.boundaryData().value().view();
    const auto [phiInternal, phiBoundaryValue, faceCells, deltaCoeffs, normals] = views(
        phi.internalVector(),
        phi.boundaryData().value(),
        mesh.boundaryMesh().faceOwners(),
        mesh.boundaryMesh().deltaCoeffs(),
        mesh.boundaryMesh().faceUnitNormals()
    );

    const auto nBoundaries = toLocalIdx(mesh.nBoundaries());
    for (localIdx patchID = 0; patchID < nBoundaries; ++patchID)
    {
        const auto [start, end] = phi.boundaryData().range(patchID);

        if (start == end)
        {
            continue;
        }

        // The normal derivative is the patch's own snGrad, deltaCoeffs*(value - cell value), as
        // OpenFOAM's gaussGrad::correctBoundaryConditions takes it from fvPatchField::snGrad().
        // That holds for every patch type: fixedGradient's value is cell + refGrad/delta, a mixed
        // patch's (inletOutlet) blends to the inlet value on inflow faces, and a symmetry plane's
        // is the tangential projection. refGrad alone is wrong for the latter two.
        parallelFor(
            exec,
            {start, end},
            NEON_LAMBDA(const localIdx i) {
                const auto owner = faceCells[i];

                // Extrapolate internal gradient
                Vec3 g = gradInternal[owner];

                // Compute snGrad
                const scalar snGrad = (phiBoundaryValue[i] - phiInternal[owner]) * deltaCoeffs[i];
                const Vec3 n = normals[i];

                // Normal reconstruction
                const scalar nDotG = n[0] * g[0] + n[1] * g[1] + n[2] * g[2];

                g += n * (snGrad - nDotG);

                gradBoundary[i] = g;
            },
            "computeGradBoundaryFixedValue"
        );
    }
}

GaussGreenGrad::GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh)
    : Base(exec, mesh),
      surfaceInterpolation_(exec, mesh, std::make_unique<Linear<scalar>>(exec, mesh, Dictionary())),
      surfaceInterpolationVec_(
          exec, mesh, std::make_unique<Linear<Vec3>>(exec, mesh, Dictionary())
      ) {};

GaussGreenGrad::GaussGreenGrad(
    const Executor& exec, const UnstructuredMesh& mesh, const Input& /* inputs */
)
    : GaussGreenGrad(exec, mesh) {};


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
    // Proc-aware calculated BCs: processor patches carry the halo-exchange BC so the corrected /
    // limitedCorrected face-normal gradient can pull the neighbour cell gradient across the rank
    // boundary with a single correctBoundaryConditions(). Physical patches stay 'calculated'.
    auto gradBCs = createCalculatedProcBCs<VolumeBoundary<Vec3>>(phi.mesh());
    VolumeField<Vec3> gradPhi = VolumeField<Vec3>(phi.exec(), "gradPhi", phi.mesh(), gradBCs);
    fill(gradPhi.internalVector(), zero<Vec3>());
    computeGrad(phi, surfaceInterpolation_, gradPhi.internalVector(), operatorScaling);
    computeBoundaryGrad(phi, gradPhi, operatorScaling);
    return gradPhi;
}

// ---- Tensor gradient implementation ----------------------------------------

KOKKOS_INLINE_FUNCTION
void atomicAddTensor(Tensor* target, sizeType row, sizeType col, scalar value)
{
    Kokkos::atomic_add(&(*target)(row, col), value);
}

KOKKOS_INLINE_FUNCTION
void atomicSubTensor(Tensor* target, sizeType row, sizeType col, scalar value)
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
            for (sizeType row = 0; row < 3; ++row)
            {
                for (sizeType col = 0; col < 3; ++col)
                {
                    const scalar c = sf[col] * faceU[row];
                    atomicAddTensor(&gT[o], row, col, c);
                    atomicSubTensor(&gT[n], row, col, c);
                }
            }
        },
        "computeGradTensorInternal"
    );

    const auto bFaceNormals = mesh.boundaryMesh().faceNormals().view();
    parallelFor(
        exec,
        {0, nBnd},
        NEON_LAMBDA(const localIdx bi) {
            const auto o = bFaceCells[bi];
            const Vec3 sf = bFaceNormals[bi];
            const Vec3 faceU = ufBound[bi];
            for (sizeType row = 0; row < 3; ++row)
            {
                for (sizeType col = 0; col < 3; ++col)
                {
                    atomicAddTensor(&gT[o], row, col, sf[col] * faceU[row]);
                }
            }
        },
        "computeGradTensorBoundary"
    );

    // Processor faces: add the proc-face flux to the owner cell's gradient, exactly as the scalar
    // computeGradScalar does. This block was MISSING from the tensor path, so the cell gradient at
    // every processor-adjacent cell omitted its proc-face contribution — corrupting the Vec3
    // corrected/limited snGrad at proc-adjacent internal faces (the uncorrected scheme, which does
    // not use the cell gradient, was unaffected). Proc faces are compressed: use the boundary
    // mesh's own faceNormals indexed by nBoundaryFaces+procFacei, NOT the OF-full SfAll.
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        const auto bSf = mesh.boundaryMesh().faceNormals().view();
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const auto bcfacei = nBnd + procFacei;
                const auto o = bFaceCells[bcfacei];
                const Vec3 sf = bSf[bcfacei];
                const Vec3 faceU = ufBound[bcfacei];
                for (sizeType row = 0; row < 3; ++row)
                {
                    for (sizeType col = 0; col < 3; ++col)
                    {
                        atomicAddTensor(&gT[o], row, col, sf[col] * faceU[row]);
                    }
                }
            },
            "computeGradTensorProcBoundary"
        );
    }

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

    const auto [UInt, UB, faceCells, deltaCoeffs, nHat] = views(
        u.internalVector(),
        u.boundaryData().value(),
        mesh.boundaryMesh().faceOwners(),
        mesh.boundaryMesh().deltaCoeffs(),
        mesh.boundaryMesh().faceUnitNormals()
    );

    for (localIdx patchID = 0; patchID < static_cast<localIdx>(offsets.size() - 1); ++patchID)
    {
        const localIdx start = offsets[static_cast<sizeType>(patchID)];
        const localIdx end = offsets[static_cast<sizeType>(patchID + 1)];
        if (start == end) continue;

        // On a symmetry plane OpenFOAM's gradient is itself a symmetry field (a constraint patch
        // keeps its type whatever type the field asks for): its boundary value is the reflection
        // average (g + R g R)/2, R = I - 2 n n, before the snGrad correction. That removes the
        // components odd under the reflection, e.g. the tangential derivatives of the normal
        // velocity. (A vector gradient needs no such step: the correction replaces its normal
        // component anyway.)
        const auto bcName = bcs[static_cast<sizeType>(patchID)].name();
        const bool symmetryPlane = bcName == "symmetry" || bcName == "symmetryPlane";

        parallelFor(
            exec,
            {static_cast<sizeType>(start), static_cast<sizeType>(end)},
            NEON_LAMBDA(const localIdx i) {
                const auto owner = faceCells[i];
                const Vec3 n = nHat[i];
                Tensor g = gTInt[owner];

                if (symmetryPlane)
                {
                    // (R g R)_ij = g_ij - 2 n_i (n.g)_j - 2 (g.n)_i n_j + 4 n_i n_j (n.g.n)
                    Vec3 ng(0, 0, 0), gn(0, 0, 0);
                    for (sizeType a = 0; a < 3; ++a)
                    {
                        for (sizeType b = 0; b < 3; ++b)
                        {
                            ng[b] += n[a] * g(a, b);
                            gn[a] += g(a, b) * n[b];
                        }
                    }
                    const scalar ngn = n[0] * gn[0] + n[1] * gn[1] + n[2] * gn[2];
                    for (sizeType a = 0; a < 3; ++a)
                    {
                        for (sizeType b = 0; b < 3; ++b)
                        {
                            const scalar rgr = g(a, b) - 2 * n[a] * ng[b] - 2 * gn[a] * n[b]
                                             + 4 * n[a] * n[b] * ngn;
                            g(a, b) = scalar(0.5) * (g(a, b) + rgr);
                        }
                    }
                }

                // The patch's own snGrad (see computeBoundaryGrad).
                const Vec3 snGrad = (UB[i] - UInt[owner]) * deltaCoeffs[i];

                // Reconstruct each row: g(row,:) += n * (snGrad[row] - n · g(row,:))
                for (sizeType row = 0; row < 3; ++row)
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
    // Proc-aware calculated BCs (see grad()): processor patches carry the halo-exchange BC so the
    // tensor gradient's neighbour value can be fetched via correctBoundaryConditions().
    auto calcBC = createCalculatedProcBCs<VolumeBoundary<Tensor>>(u.mesh());
    VolumeField<Tensor> gradU(u.exec(), "gradU", u.mesh(), calcBC);
    fill(gradU.internalVector(), zero<Tensor>());

    computeGradTensor(u, surfaceInterpolationVec_, gradU.internalVector(), operatorScaling);
    computeBoundaryGradTensor(u, gradU);
    return gradU;
}

} // namespace NeoN
