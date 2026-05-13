// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <tuple>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/fields/boundaryData.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/basicGeometryScheme.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred
{

BasicGeometryScheme::BasicGeometryScheme(const UnstructuredMesh& mesh)
    : GeometrySchemeFactory(mesh), mesh_(mesh)
{}

namespace
{

/**
 * @brief Build the (start, end) ranges of all processor-boundary patches in the
 *        boundaryData layout (i.e. into a vector sized nBoundaryFaces + nProcBoundaryFaces),
 *        in MESH-BOUNDARY ORDER, paired with their target ranks.
 *
 * Processor patches are the trailing patches in the boundary mesh; their offsets
 * already point into the boundary-data layout where the proc tail begins at
 * `boundaryMesh.nBoundaryFaces()`.
 *
 * communicateBoundaryData uses targetRanks to compute MPI Alltoallv
 * displacements per-rank, so mesh-order is preserved end-to-end (which is
 * what downstream consumers like setProcBoundarySparsityPattern expect).
 */
std::pair<std::vector<std::pair<localIdx, localIdx>>, std::vector<int>>
collectProcPatchOffsets(const UnstructuredMesh& mesh)
{
    std::vector<std::pair<localIdx, localIdx>> procPatchOffset;
    std::vector<int> targetRanks;
    const auto& patchOffsets = mesh.boundaryMesh().offset();
    const auto& nbrRanks = mesh.boundaryMesh().neighbourRank();
    const auto totalPatches = mesh.boundaryMesh().nBoundaries();
    const auto procPatchCount = mesh.boundaryMesh().nProcBoundaryPatches();
    if (procPatchCount == 0)
    {
        return {procPatchOffset, targetRanks};
    }
    const auto firstProcPatch = totalPatches - procPatchCount;

    procPatchOffset.reserve(static_cast<std::size_t>(procPatchCount));
    targetRanks.reserve(static_cast<std::size_t>(procPatchCount));
    for (localIdx p = firstProcPatch; p < totalPatches; ++p)
    {
        const auto procIdx = p - firstProcPatch;
        procPatchOffset.emplace_back(patchOffsets[p], patchOffsets[p + 1]);
        targetRanks.push_back(static_cast<int>(nbrRanks[procIdx]));
    }
    return {procPatchOffset, targetRanks};
}

/**
 * @brief Exchange the local owner-to-face orthogonal distance across processor
 *        patches via MPI_Alltoallv.
 *
 * Returns a `Vector<scalar>` sized like `boundaryData().value()` (i.e.
 * `nBoundaryFaces + nProcBoundaryFaces`). Before the exchange the proc-tail
 * entries hold this rank's `|n · (cf - c_own)|` (the local owner-to-face distance
 * projected onto the face normal). After the exchange they hold the matching
 * neighbour rank's local distance for the same physical face — i.e. the local
 * `d_neighbour`. Physical-boundary entries are left at zero.
 *
 * If the mesh is not distributed (no processor patches) the data is left
 * untouched and no MPI call is made.
 */
Vector<scalar> exchangeProcOwnerDistance(const Executor& exec, const UnstructuredMesh& mesh)
{
    const auto nBoundaryFaces = mesh.boundaryMesh().nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.boundaryMesh().nProcBoundaryFaces();
    const auto totalBoundary = nBoundaryFaces + nProcBoundaryFaces;

    Vector<scalar> dExchange(exec, totalBoundary, 0.0);

    if (nProcBoundaryFaces == 0)
    {
        return dExchange;
    }

    const auto cellCentre = mesh.cellCentres().view();
    const auto bcCf = mesh.boundaryMesh().cf().view();
    const auto bcSf = mesh.boundaryMesh().sf().view();
    const auto bcMagSf = mesh.boundaryMesh().magSf().view();
    const auto surfFaceCells = mesh.boundaryMesh().faceCells().view();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto totalFaces = mesh.nTotalFaces();
    auto dExchangeV = dExchange.view();

    // Fill local d_own at the proc-tail positions.
    //
    // mesh.faceCentres()/faceAreas()/magFaceAreas() are sized over OpenFOAM's
    // full face list (which includes empty patches like defaultFaces); indexing
    // them with the compressed proc-face index reads the wrong empty-patch face.
    // bm.cf()/sf()/magSf() are in the compressed boundary-tail layout that
    // matches `bcfacei = facei - nInternalFaces`, so use those instead.
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto bfacei = facei - nInternalFaces;
            const auto own = surfFaceCells[bfacei];
            const Vec3 cellToFace = bcCf[bfacei] - cellCentre[own];
            const Vec3 faceNormal = (1.0 / bcMagSf[bfacei]) * bcSf[bfacei];
            // Owner side: outward normal from own cell, distance is positive.
            dExchangeV[bfacei] = Kokkos::abs(static_cast<scalar>(faceNormal & cellToFace));
        },
        "exchangeProcOwnerDistance::fillLocal"
    );

    auto [procPatchOffset, targetRanks] = collectProcPatchOffsets(mesh);
    if (procPatchOffset.empty())
    {
        return dExchange;
    }

    // FIXME share commPattern with VolumeField::correctBoundaryConditions to avoid
    // recomputing it once per geometry update. For now match the pattern used there.
    auto commPattern = computeCommunicationPattern(mesh);
    communicateBoundaryData(commPattern, procPatchOffset, targetRanks, dExchange);

    return dExchange;
}

} // namespace

void BasicGeometryScheme::updateWeights(const Executor& exec, SurfaceField<scalar>& weights)
{
    const auto owner = mesh_.faceOwner().view();
    const auto neighbour = mesh_.faceNeighbour().view();

    const auto cf = mesh_.faceCentres().view();
    const auto c = mesh_.cellCentres().view();
    const auto sf = mesh_.faceAreas().view();
    const auto bcCf = mesh_.boundaryMesh().cf().view();
    const auto bcSf = mesh_.boundaryMesh().sf().view();
    const auto bcMagSf = mesh_.boundaryMesh().magSf().view();


    const auto [weightS, weightB] = views(weights.internalVector(), weights.boundaryData().value());

    const auto nInternalFaces = mesh_.nInternalFaces();
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto totalFaces = mesh_.nTotalFaces();

    // NF_ASSERT(dstS.size() == ownerS.size(), "Inconsistent size");
    // NF_ASSERT(dstS.size() == neighS.size(), "Inconsistent size");
    // NF_ASSERT(dstS.size() == weightS.size(), "Inconsistent size");

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            scalar sfdOwn = Kokkos::abs(sf[facei] & (cf[facei] - c[owner[facei]]));
            scalar sfdNei = Kokkos::abs(sf[facei] & (c[neighbour[facei]] - cf[facei]));

            if (Kokkos::abs(sfdOwn + sfdNei) > ROOTVSMALL)
            {
                weightS[facei] = sfdNei / (sfdOwn + sfdNei);
            }
            else
            {
                weightS[facei] = 0.5;
            }
        },
        "basicGeometricScheme::updateWeightsInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + nBoundaryFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto bcfacei = facei - nInternalFaces;
            weightS[facei] = 1.0;
            weightB[bcfacei] = 1.0;
        },
        "basicGeometricScheme::updateWeightsBoundary"
    );

    // Processor boundary: linear interpolation weight between the local owner cell
    // and the ghost cell on the neighbouring rank.
    //
    // For an internal face the weight is
    //   w = sfdNei / (sfdOwn + sfdNei)    with    sfdOwn = |Sf · (cf - c_own)|,
    //                                              sfdNei = |Sf · (c_nei - cf)|,
    // matching `phi_f = w * phi_own + (1 - w) * phi_nei`.
    //
    // For processor faces c_nei lives on the neighbour rank. To mirror
    // OpenFOAM's processorFvPatch::makeWeights (which is consistent on both sides
    // of the cut and produces a symmetric Laplacian matrix at proc boundaries) we
    // exchange the local owner-to-face distance via MPI and form
    //   w = d_nei / (d_own + d_nei)
    // where d_own and d_nei are the projections of (cf - c_own) onto the face
    // normal on each side. On a uniform decomposition this collapses to 0.5.
    //
    // Skip the exchange and the loop entirely on a serial / non-distributed mesh
    // — `nProcBoundaryFaces == 0` ⇒ no proc faces, no MPI work, and no Vector
    // allocation that would otherwise happen for nothing.
    if (mesh_.boundaryMesh().nProcBoundaryFaces() > 0)
    {
        auto dNeighbourBoundary = exchangeProcOwnerDistance(exec, mesh_);
        auto dNeighbourBoundaryV = dNeighbourBoundary.view();
        const auto procFaceCells = mesh_.boundaryMesh().faceCells().view();
        parallelFor(
            exec,
            {nInternalFaces + nBoundaryFaces, totalFaces},
            NEON_LAMBDA(const localIdx facei) {
                const auto bcfacei = facei - nInternalFaces;
                const auto own = procFaceCells[bcfacei];
                const Vec3 cellToFace = bcCf[bcfacei] - c[own];
                const scalar magSf = bcMagSf[bcfacei];
                const Vec3 faceNormal =
                    (magSf > ROOTVSMALL ? scalar(1) / magSf : scalar(0)) * bcSf[bcfacei];
                const scalar dOwn = Kokkos::abs(static_cast<scalar>(faceNormal & cellToFace));
                const scalar dNei = dNeighbourBoundaryV[bcfacei];
                const scalar denom = dOwn + dNei;
                const scalar w = (denom > ROOTVSMALL) ? (dNei / denom) : scalar(0.5);
                weightS[facei] = w;
                weightB[bcfacei] = w;
            },
            "basicGeometricScheme::updateWeightsProcBoundary"
        );
    }
}

void BasicGeometryScheme::updateDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& deltaCoeffs)
{
    const auto [owner, neighbour, surfFaceCells] =
        views(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());


    const auto [cf, cellCentre] = views(mesh_.faceCentres(), mesh_.cellCentres());

    auto deltaCoeff = deltaCoeffs.internalVector().view();

    parallelFor(
        exec,
        {0, mesh_.nInternalFaces()},
        NEON_LAMBDA(const localIdx facei) {
            Vec3 cellToCellDist = cellCentre[neighbour[facei]] - cellCentre[owner[facei]];
            deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
        },
        "basicGeometricScheme::updateDeltaCoeffsInternal"
    );

    const auto nInternalFaces = mesh_.nInternalFaces();
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto totalFaces = mesh_.nTotalFaces();

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + nBoundaryFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = surfFaceCells[facei - nInternalFaces];
            Vec3 cellToCellDist = cf[facei] - cellCentre[own];

            deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
        },
        "basicGeometricScheme::updateDeltaCoeffsBoundary"
    );

    // GEO-01 fix: proc-face deltaCoeffs uses the full cell-to-cell distance
    // (d_own + d_nei) across the decomposition cut, mirroring
    // updateNonOrthDeltaCoeffs. d_nei is obtained via MPI exchange of the
    // owner's face-normal-projected distance from the neighbour rank.
    // bm.cf()/sf()/magSf() are in compressed boundary-tail indexing matching
    // bcfacei; mesh_.faceCentres()/faceAreas()/magFaceAreas() are sized over
    // OpenFOAM's full face list (incl. empty patches) and would read the
    // wrong face here.
    if (mesh_.boundaryMesh().nProcBoundaryFaces() > 0)
    {
        auto dNeighbourBoundary = exchangeProcOwnerDistance(exec, mesh_);
        auto dNeighbourBoundaryV = dNeighbourBoundary.view();
        const auto bcCf = mesh_.boundaryMesh().cf().view();
        const auto bcSf = mesh_.boundaryMesh().sf().view();
        const auto bcMagSf = mesh_.boundaryMesh().magSf().view();
        parallelFor(
            exec,
            {nInternalFaces + nBoundaryFaces, totalFaces},
            NEON_LAMBDA(const localIdx facei) {
                const auto bcfacei = facei - nInternalFaces;
                const auto own = surfFaceCells[bcfacei];
                const Vec3 cellToFace = bcCf[bcfacei] - cellCentre[own];
                const Vec3 faceNormal = (1.0 / bcMagSf[bcfacei]) * bcSf[bcfacei];
                const scalar dOwn = Kokkos::abs(static_cast<scalar>(faceNormal & cellToFace));
                const scalar dNei = dNeighbourBoundaryV[bcfacei];
                const scalar dCellToCell = dOwn + dNei;
                deltaCoeff[facei] =
                    (dCellToCell > ROOTVSMALL) ? scalar(1) / dCellToCell : scalar(0);
            },
            "basicGeometricScheme::updateDeltaCoeffsProcBoundary"
        );
    }
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    const auto [owner, neighbour, surfFaceCells] =
        views(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());

    const auto [cf, cellCentre, faceAreaVec3, faceArea] =
        views(mesh_.faceCentres(), mesh_.cellCentres(), mesh_.faceAreas(), mesh_.magFaceAreas());

    auto nonOrthDeltaCoeff = nonOrthDeltaCoeffs.internalVector().view();
    fill(nonOrthDeltaCoeffs.internalVector(), 0.0);

    const auto nInternalFaces = mesh_.nInternalFaces();
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto totalFaces = mesh_.nTotalFaces();

    NeoN::mpi::Environment mpiEnviron;
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            Vec3 cellToCellDist = cellCentre[neighbour[facei]] - cellCentre[owner[facei]];
            Vec3 faceNormal = 1 / faceArea[facei] * faceAreaVec3[facei];
            scalar orthoDist = faceNormal & cellToCellDist;
            nonOrthDeltaCoeff[facei] = 1.0 / Kokkos::max(orthoDist, 0.05 * mag(cellToCellDist));
        },
        "basicGeometricScheme::updateNonOrthDeltaCoeffsInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + nBoundaryFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = surfFaceCells[facei - nInternalFaces];
            Vec3 cellToCellDist = cf[facei] - cellCentre[own];
            Vec3 faceNormal = 1 / faceArea[facei] * faceAreaVec3[facei];
            scalar orthoDist = faceNormal & cellToCellDist;
            nonOrthDeltaCoeff[facei] = 1.0 / Kokkos::max(orthoDist, 0.05 * mag(cellToCellDist));
        },
        "basicGeometricScheme::updateNonOrthDeltaCoeffsBoundary"
    );

    // Processor boundary nonOrthDeltaCoeffs.
    //
    // The Laplacian assembly at processor patches uses this coefficient as the
    // inverse cell-to-cell distance across the proc face. To make the assembled
    // matrix symmetric across the cut on non-uniform meshes we need
    //   1 / (d_own + d_nei)
    // computed identically on both ranks, where d_nei is the matching neighbour-
    // side owner-to-face distance fetched via MPI. The previous formulation
    //   0.5 / d_own
    // was only correct on uniform decompositions (d_own == d_nei) and produced an
    // asymmetric Laplacian — and a visible pressure dipole at proc boundaries —
    // on graded meshes.
    //
    // Skip on serial / non-distributed meshes (no proc faces ⇒ nothing to do).
    if (mesh_.boundaryMesh().nProcBoundaryFaces() > 0)
    {
        auto dNeighbourBoundary = exchangeProcOwnerDistance(exec, mesh_);
        auto dNeighbourBoundaryV = dNeighbourBoundary.view();
        // bm.cf()/sf()/magSf() are in compressed boundary-tail indexing matching
        // bcfacei. mesh_.faceCentres()/faceAreas()/magFaceAreas() are sized over
        // OpenFOAM's full face list (incl. empty patches) and would read the
        // wrong face here.
        const auto bcCf = mesh_.boundaryMesh().cf().view();
        const auto bcSf = mesh_.boundaryMesh().sf().view();
        const auto bcMagSf = mesh_.boundaryMesh().magSf().view();
        parallelFor(
            exec,
            {nInternalFaces + nBoundaryFaces, totalFaces},
            NEON_LAMBDA(const localIdx facei) {
                const auto bcfacei = facei - nInternalFaces;
                const auto own = surfFaceCells[bcfacei];
                const Vec3 cellToFace = bcCf[bcfacei] - cellCentre[own];
                const Vec3 faceNormal = (1.0 / bcMagSf[bcfacei]) * bcSf[bcfacei];
                const scalar dOwn = Kokkos::abs(static_cast<scalar>(faceNormal & cellToFace));
                const scalar dNei = dNeighbourBoundaryV[bcfacei];
                const scalar dCellToCell = dOwn + dNei;
                // Cell-to-cell vector approximation for the floor (avoid divide-by-zero
                // / very small denominators in pathological cases).
                const Vec3 approxCellToCell = cellToFace + faceNormal * dNei;
                nonOrthDeltaCoeff[facei] =
                    1.0 / Kokkos::max(dCellToCell, scalar(0.05) * mag(approxCellToCell));
            },
            "basicGeometricScheme::updateNonOrthDeltaCoeffsProcBoundary"
        );
    }
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<Vec3>& nonOrthDeltaCoeffs
)
{
    NF_ERROR_EXIT("Not implemented");
}

} // namespace NeoN
