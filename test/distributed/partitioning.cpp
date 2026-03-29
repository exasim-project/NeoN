// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace dsl = NeoN::dsl;


namespace NeoN
{

/** @brief helper function to set the processor boundaries of a distributed field */
template<typename BoundaryType>
auto setProcessorBoundaryHelper(std::vector<BoundaryType> bcs, size_t rank)
{
    return bcs;
}

/** @brief helper function given a 1D uniform mesh and a rank it will return the part of the mesh
 owned by this rank */
template<typename FieldType>
FieldType
partitionVolField(FieldType field, auto& mesh, auto bcs, NeoN::mpi::Environment mpiEnviron)
{
    localIdx localCells = mesh.nCells();
    localIdx firstCell = 0;
    localIdx lastCell = localCells;

    if (mpiEnviron.rank() == 0)
    {
        lastCell = localCells;
    }
    if (mpiEnviron.rank() == 1)
    {
        firstCell = localCells;
        lastCell = localCells + localCells;
    }
    if (mpiEnviron.rank() == 2)
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
    localIdx localCells = mesh.nCells();  // 4
    localIdx localFaces = localCells - 1; // 3
    localIdx firstFace = 0;
    localIdx lastFace = localFaces;

    localIdx leftBoundaryFace = 0;
    localIdx rightBoundaryFace = 0;

    scalar signLeft = 1.0;
    scalar signRight = 1.0;

    // [ 0 | 1 | 2 | 3 ][ 4 | 5 | 6 | 7 ][ 8 | 9 | 10 | 11 ]
    // 11  0   1   2   3    4   5   6   7    8   9   10   12

    if (mpiEnviron.rank() == 0)
    {
        lastFace = localFaces + 1;             // 4
        leftBoundaryFace = 3 * localFaces + 2; // 11
        rightBoundaryFace = localFaces;        // 3
    }
    if (mpiEnviron.rank() == 1)
    {
        firstFace = localFaces + 1;            // 4  last face rank 0
        lastFace = firstFace + localFaces + 1; // 8

        leftBoundaryFace = localFaces;                     // should be 3
        rightBoundaryFace = leftBoundaryFace + localCells; // should 3 + 4

        // new face has different direction compared to unpartitioned case
        // signRight = -1.0;
        if (flip)
        {
            signLeft = -1.0;
        }
    }
    if (mpiEnviron.rank() == 2)
    {
        firstFace = localCells + localCells; // 8 last face rank 1
        lastFace = firstFace + localFaces + 1;

        leftBoundaryFace = 2 * localFaces + 1;                 // 7
        rightBoundaryFace = leftBoundaryFace + localCells + 1; // 12

        // new face has different direction compared to unpartitioned case
        if (flip)
        {
            signLeft = -1.0;
        }
    }

    FieldType ret = {field.exec(), field.name + "Part", mesh, bcs};

    // NOTE last two values are boundaries and are overwritten next
    auto internalVector = take(field.internalVector(), firstFace, lastFace + 1);
    // value lastFace  and lastFace+1 are incorrect
    // value lastFace is the left boundary
    // and lastFace + 1 is at the new boundary so it should be lastFace

    auto outV = internalVector.view();
    auto inV = field.internalVector().view();


    // set left boundary face
    NeoN::parallelFor(
        // lastface
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx i) { outV[localFaces] = signLeft * inV[leftBoundaryFace]; },
        "copyMap"
    );

    NeoN::parallelFor(
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx i) {
            outV[localFaces + 1] = signRight * inV[rightBoundaryFace];
        },
        "copyMap"
    );

    NF_ASSERT(ret.internalVector().size() == internalVector.size(), "different size");
    ret.internalVector() = internalVector;
    return ret;
}

TEST_CASE("Distributed")
{
    // start with non distributed setup
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // global number of cells
    auto nCells = 12;
    NeoN::mpi::Environment mpiEnviron;
    auto meshPart = create1DUniformMeshPart(exec, nCells / mpiEnviron.sizeRank(), mpiEnviron);

    SECTION("Has correct partitioned 1d mesh" + execName)
    {
        REQUIRE(meshPart.nCells() == nCells / mpiEnviron.sizeRank());

        if (mpiEnviron.rank() == 1)
        {
            REQUIRE(meshPart.nBoundaryFaces() == 0);
        }
        else
        {
            REQUIRE(meshPart.nBoundaryFaces() == 1);
        }
    }
}

}
