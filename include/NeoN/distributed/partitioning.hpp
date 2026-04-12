// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/label.hpp"

namespace NeoN
{

/** @brief helper function given a 1D uniform mesh and a rank it will return the part of the mesh
 owned by this rank */
template<typename FieldType>
FieldType
partitionVolField(FieldType field, const auto& mesh, auto bcs, NeoN::mpi::Environment mpiEnviron)
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

/** @brief helper function given a 1D uniform mesh and a rank it will return the part of the mesh
 owned by this rank */
template<typename FieldType>
FieldType partitionSurfaceField(
    FieldType field, auto& mesh, auto bcs, NeoN::mpi::Environment mpiEnviron, bool flip = false
)
{
    auto exec = mesh.exec();
    localIdx localCells = mesh.nCells();         // 4
    localIdx localFaces = mesh.nInternalFaces(); // 3
    localIdx firstFace = 0;
    localIdx lastFace = localFaces;

    //
    localIdx firstBoundaryFace = 0;
    localIdx secondBoundaryFace = 0;

    scalar signLeft = 1.0;
    scalar signRight = 1.0;

    // FIXME this only works for 3 ranks
    if (mpiEnviron.rank() == 0)
    {
        // first boundary face is the left boundary remaining local
        firstBoundaryFace = 3 * localFaces + 2; // 11
        secondBoundaryFace = localFaces;        // 3

        if (flip)
        {
            // signLeft = -1.0;
            signRight = -1.0;
        }
    }
    if (mpiEnviron.rank() == 1)
    {
        firstFace = localFaces + 1; // 4  last face rank 0

        firstBoundaryFace = localFaces;                      // should be 3
        secondBoundaryFace = firstBoundaryFace + localCells; // should 3 + 4

        // new face has different direction compared to unpartitioned case
        // signRight = -1.0;
        if (flip)
        {
            signLeft = -1.0;
            // signRight = -1.0;
        }
    }
    if (mpiEnviron.rank() == 2)
    {
        firstFace = localCells + localCells; // 8 last face rank 1

        // first boundary face is the right boundary remaining local
        // firstBoundaryFace = 2 * localFaces + 1;                 // 7
        firstBoundaryFace = 3 * localFaces + 3;  // 11
        secondBoundaryFace = 2 * localFaces + 1; // firstBoundaryFace + localCells + 1; // 12

        // NOTE FIXME signRight is flipped since
        // new face has different direction compared to unpartitioned case
        if (flip)
        {
            // signLeft = -1.0;
            signRight = -1.0;
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
