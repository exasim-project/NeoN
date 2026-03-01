// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/partition/partitionMesh.hpp"
#include "NeoN/mesh/unstructured/partition/extractSubMesh.hpp"

// --- Cycle 1: trivial single-part case (no METIS needed) ---

TEST_CASE("partitionMesh - nParts==1 assigns all cells to part 0")
{
    auto exec = NeoN::SerialExecutor {};
    auto mesh = NeoN::createUniform2DGrid(exec, 2, 2); // 4 cells
    auto cellPart = NeoN::partition::partitionMesh(mesh, 1);
    REQUIRE(static_cast<NeoN::localIdx>(cellPart.size()) == mesh.nCells());
    for (auto p : cellPart)
    {
        REQUIRE(p == 0);
    }
}

// --- Cycle 2: METIS balanced assignment ---

TEST_CASE("partitionMesh - balanced cell assignment")
{
    auto exec = NeoN::SerialExecutor {};
    auto mesh = NeoN::createUniform2DGrid(exec, 4, 4); // 16 cells
    auto cellPart = NeoN::partition::partitionMesh(mesh, 4);
    REQUIRE(static_cast<NeoN::localIdx>(cellPart.size()) == mesh.nCells());
    for (auto p : cellPart)
    {
        REQUIRE(p >= 0);
        REQUIRE(p < 4);
    }
    std::vector<int> counts(4, 0);
    for (auto p : cellPart)
        counts[static_cast<std::size_t>(p)]++;
    for (auto c : counts)
    {
        REQUIRE(c >= 2);
        REQUIRE(c <= 8);
    }
}

// --- Cycle 3: sub-mesh cell count ---

TEST_CASE("extractSubMesh - total cell count preserved")
{
    auto exec = NeoN::SerialExecutor {};
    auto mesh = NeoN::createUniform2DGrid(exec, 4, 4); // 16 cells
    auto cellPart = NeoN::partition::partitionMesh(mesh, 4);
    NeoN::localIdx total = 0;
    for (int p = 0; p < 4; p++)
    {
        total += NeoN::partition::extractSubMesh(mesh, cellPart, p).nCells();
    }
    REQUIRE(total == mesh.nCells());
}
