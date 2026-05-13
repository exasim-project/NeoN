// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/label.hpp"

namespace NeoN
{

/** @brief helper function given a 1D uniform mesh and a rank it will return the part of the mesh
 owned by this rank */
template<typename FieldType, typename MeshType, typename BcType>
FieldType partitionVolField(
    FieldType field, const MeshType& mesh, BcType bcs, NeoN::mpi::Environment mpiEnviron
)
{
    localIdx localCells = mesh.nCells();
    localIdx firstCell = 0;
    localIdx lastCell = localCells;

    if (mpiEnviron.rank() == 0)
    {
        lastCell = localCells;
    }
    if (mpiEnviron.rank() >= 1 && mpiEnviron.rank() != mpiEnviron.sizeRank() - 1)
    {
        firstCell = localCells;
        lastCell = localCells + localCells;
    }

    if (mpiEnviron.rank() == mpiEnviron.sizeRank() - 1)
    {
        firstCell = localCells + localCells;
        lastCell = localCells + localCells + localCells;
    }

    auto internalVector = take(field.internalVector(), firstCell, lastCell);

    return {field.exec(), field.name + "Part", mesh, internalVector, bcs};
}

/** @brief Partition a SurfaceField of a 1D uniform mesh into the slice owned by
 * `mpiEnviron.rank()`.
 *
 * For a 1D uniform mesh with R = mpiEnviron.sizeRank() ranks and localCells = mesh.nCells()
 * cells per rank (localFaces = localCells - 1 internal faces per rank), this function extracts
 * the contiguous slice [firstFace, firstFace + localFaces + 2) from the full surface field
 * and patches in the two boundary face values (left/right) that the slice cannot inherit
 * directly because they belong to a neighbouring rank or to the global domain boundary.
 *
 * Closed-form N-rank formulas (see .planning/phases/02-linear-system-correctness/
 * 02-RESEARCH.md NRANK-01 section for derivation):
 *
 *   firstFace             = r * (localFaces + 1)              for all r
 *
 *   First rank (r == 0):
 *     firstBoundaryFace   = R * (localFaces + 1) - 1          (global left domain boundary)
 *     secondBoundaryFace  = localFaces                        (right proc-face)
 *
 *   Middle rank (0 < r < R - 1):
 *     firstBoundaryFace   = r * (localFaces + 1) - 1          (left proc-face)
 *     secondBoundaryFace  = r * (localFaces + 1) + localFaces (right proc-face)
 *
 *   Last rank (r == R - 1):
 *     firstBoundaryFace   = R * (localFaces + 1)              (global right domain boundary)
 *     secondBoundaryFace  = (R - 1) * (localFaces + 1) - 1    (left proc-face)
 *
 * Hand-verified oracles (see .planning/phases/02-linear-system-correctness/02-02-SUMMARY.md
 * for the full R=2/3/4 derivation tables):
 *
 *   R=3 (automated test, partitioning.cpp:218-230) with internalVector = {1..11, 20, 30}:
 *     Rank 0 -> {1, 2, 3, 20, 4}
 *     Rank 1 -> {5, 6, 7,  4, 8}
 *     Rank 2 -> {9,10,11, 30, 8}
 *
 *   R=2 (hand-verified) with internalVector = {1, 2, 3, 20, 30}, localCells=2:
 *     Rank 0 -> {1, 20, 2}
 *     Rank 1 -> {3, 30, 2}
 *
 *   R=4 (hand-verified) with internalVector = {1, 2, 3, 4, 5, 6, 7, 40, 50}, localCells=2:
 *     Rank 0 -> {1, 40, 2}
 *     Rank 1 -> {3,  2, 4}
 *     Rank 2 -> {5,  4, 6}
 *     Rank 3 -> {7, 50, 6}
 *
 * Sign-flip rule when `flip = true`:
 *   First rank: no flip — rank 0 is the global owner of its proc face (Sf_local = Sf_global)
 *   Last rank:  signRight = -1.0  (left proc face stored at outV[localFaces+1])
 *   Middle:     signLeft  = -1.0  (left proc face stored at outV[localFaces])
 *
 * @tparam FieldType  A SurfaceField<T> template instantiation
 * @tparam MeshType   An UnstructuredMesh template instantiation
 * @tparam BcType     A boundary-condition list type
 *
 * @param field      Full unpartitioned surface field on the global mesh
 * @param mesh       Local (per-rank) mesh slice produced by create1DUniformMeshPart
 * @param bcs        Boundary conditions for the partitioned surface field
 * @param mpiEnviron MPI environment providing rank() and sizeRank()
 * @param flip       When true, flips the sign of one proc-face per rank (see "Sign-flip rule")
 *
 * @return A surface field containing this rank's slice with proc-face slots populated.
 */
template<typename FieldType, typename MeshType, typename BcType>
FieldType partitionSurfaceField(
    FieldType field,
    MeshType& mesh,
    BcType bcs,
    NeoN::mpi::Environment mpiEnviron,
    bool flip = false
)
{
    auto exec = mesh.exec();
    localIdx localCells = mesh.nCells();
    localIdx localFaces = mesh.nInternalFaces();
    localIdx firstFace = 0;
    localIdx lastFace = localFaces;

    localIdx firstBoundaryFace = 0;
    localIdx secondBoundaryFace = 0;

    scalar signLeft = 1.0;
    scalar signRight = 1.0;

    // N-rank generic formulas (replaces the rank==0/1/2 if/else chain).
    // Derivation and R=2/3/4 hand-verification: see
    //   .planning/phases/02-linear-system-correctness/02-02-SUMMARY.md
    //   .planning/phases/02-linear-system-correctness/02-RESEARCH.md (NRANK-01 section).
    // 3-rank oracle (src/NeoN/test/distributed/partitioning.cpp:218-230) preserved.
    const localIdx r = mpiEnviron.rank();
    const localIdx R = mpiEnviron.sizeRank();

    firstFace = r * (localFaces + 1);

    if (r == 0)
    {
        // First rank: left boundary = global left domain boundary; right boundary = right
        // proc-face. Rank 0 is the global owner of its proc face (Sf_local = Sf_global),
        // so no sign correction is needed regardless of the flip flag.
        firstBoundaryFace = R * (localFaces + 1) - 1;
        secondBoundaryFace = localFaces;
    }
    else if (r == R - 1)
    {
        // Last rank: left boundary = left proc-face; right boundary = global right domain boundary.
        firstBoundaryFace = R * (localFaces + 1);
        secondBoundaryFace = (R - 1) * (localFaces + 1) - 1;
        if (flip)
        {
            signRight = -1.0;
        }
    }
    else
    {
        // Middle rank: both boundaries are proc-faces.
        firstBoundaryFace = r * (localFaces + 1) - 1;
        secondBoundaryFace = r * (localFaces + 1) + localFaces;
        if (flip)
        {
            signLeft = -1.0;
        }
    }

    lastFace = firstFace + localFaces + 2; // two extra faces for boundary + 1

    // surface field contains all faces
    FieldType ret = {field.exec(), field.name + "Part", mesh, bcs};

    NF_ASSERT(lastFace - firstFace == mesh.nTotalFaces(), "different size");

    // NOTE last two values are boundaries and are overwritten next
    auto internalVector = take(field.internalVector(), firstFace, lastFace);
    // value lastFace  and lastFace+1 are incorrect
    // value lastFace is the left boundary
    // and lastFace + 1 is at the new boundary so it should be lastFace

    auto outV = internalVector.view();
    auto inV = field.internalVector().view();

    // set first boundary face
    NeoN::parallelFor(
        // lastface
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx i) { outV[localFaces] = signLeft * inV[firstBoundaryFace]; },
        "copyMap"
    );

    // set second boundary face
    NeoN::parallelFor(
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx i) {
            outV[localFaces + 1] = signRight * inV[secondBoundaryFace];
        },
        "copyMap"
    );


    NF_ASSERT(ret.internalVector().size() == internalVector.size(), "different size");
    ret.internalVector() = internalVector;
    return ret;
}
}
