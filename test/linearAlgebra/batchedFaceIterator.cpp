// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include <set>

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace NeoN
{

namespace
{

// Checks that segments[c]..segments[c+1] partition [0, nFaces) and that no cell appears twice in
// one batch. ownerOf(face) and neighbourOf(face) give the cells a face writes to.
template<typename OwnerOf, typename NeighbourOf>
void requireValidBatches(
    const SegmentedVector<localIdx, localIdx>& batched,
    localIdx nBatches,
    localIdx nFaces,
    OwnerOf ownerOf,
    NeighbourOf neighbourOf
)
{
    const auto faces = batched.values().copyToHost();
    const auto segments = batched.segments().copyToHost();
    const auto [facesV, segmentsV] = views(faces, segments);

    REQUIRE(segments.size() == static_cast<sizeType>(nBatches) + 1);
    REQUIRE(segmentsV[0] == 0);
    REQUIRE(segmentsV[nBatches] == nFaces);

    std::vector<int> seen(static_cast<sizeType>(nFaces), 0);
    for (localIdx c = 0; c < nBatches; ++c)
    {
        REQUIRE(segmentsV[c] < segmentsV[c + 1]); // every batch is used
        std::set<localIdx> cells;
        for (localIdx i = segmentsV[c]; i < segmentsV[c + 1]; ++i)
        {
            const auto face = facesV[i];
            ++seen[static_cast<sizeType>(face)];
            REQUIRE(cells.insert(ownerOf(face)).second);
            REQUIRE(cells.insert(neighbourOf(face)).second);
        }
    }
    REQUIRE(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));
}

} // namespace

TEST_CASE("BatchedFaceIterator")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create3DUniformMesh(exec, 3, 4, 5);
    const auto nInternalFaces = static_cast<localIdx>(mesh.nInternalFaces());

    la::BatchedFaceIterator iterator;
    REQUIRE(iterator.getBatchedFaceData() == nullptr);
    REQUIRE(iterator.name() == "BatchedFace");

    iterator.setComputeBatchedFaceData(mesh);
    auto data = iterator.getBatchedFaceData();
    REQUIRE(data != nullptr);

    const auto owner = mesh.faceOwners().copyToHost();
    const auto neighbour = mesh.faceNeighbors().copyToHost();
    const auto [ownerV, neighbourV] = views(owner, neighbour);

    SECTION("size is the number of internal-face batches " + execName)
    {
        REQUIRE(iterator.size() == static_cast<sizeType>(data->nBatches));
        // hex cells have six faces: the greedy colouring needs at most 2 * 6 - 1 batches
        REQUIRE(data->nBatches >= 3);
        REQUIRE(data->nBatches <= 11);
    }

    SECTION("internal faces of one batch share no cell " + execName)
    {
        requireValidBatches(
            data->batchFaces,
            data->nBatches,
            nInternalFaces,
            [&](localIdx face) { return static_cast<localIdx>(ownerV[face]); },
            [&](localIdx face) { return static_cast<localIdx>(neighbourV[face]); }
        );
    }
}

}
