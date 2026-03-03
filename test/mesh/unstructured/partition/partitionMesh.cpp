// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/partition/partitionMesh.hpp"
#include "NeoN/mesh/unstructured/partition/extractSubMesh.hpp"

#include <set>

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

// --- Cycle 3: per-neighbor proc boundary patch names ---

TEST_CASE("extractSubMesh - proc patches named per neighbor partition")
{
    auto exec = NeoN::SerialExecutor {};
    auto mesh = NeoN::createUniform2DGrid(exec, 4, 4); // 16 cells
    auto cellPart = NeoN::partition::partitionMesh(mesh, 4);

    for (int p = 0; p < 4; ++p)
    {
        auto sub = NeoN::partition::extractSubMesh(mesh, cellPart, p);
        auto& patchNames =
            *sub.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");

        for (const auto& name : patchNames)
        {
            if (name.substr(0, 4) == "proc")
            {
                // Must match pattern "proc<X>to<Y>" where X == p
                REQUIRE(name.find("proc" + std::to_string(p) + "to") == 0);
                // Must NOT use the old lumped name
                REQUIRE(name.find("procBoundary_") == std::string::npos);
            }
        }
    }
}

// --- Cycle 4: global cell IDs stored in stencilDB ---

TEST_CASE("extractSubMesh - stores global cell IDs in stencilDB")
{
    auto exec = NeoN::SerialExecutor {};
    auto mesh = NeoN::createUniform2DGrid(exec, 4, 4); // 16 cells
    auto cellPart = NeoN::partition::partitionMesh(mesh, 4);

    std::vector<bool> globalSeen(16, false);
    for (int p = 0; p < 4; ++p)
    {
        auto sub = NeoN::partition::extractSubMesh(mesh, cellPart, p);
        REQUIRE(sub.stencilDB().contains("partition::globalCellIds"));

        auto& globalIds = *sub.stencilDB().get<std::shared_ptr<std::vector<NeoN::localIdx>>>(
            "partition::globalCellIds"
        );

        REQUIRE(static_cast<NeoN::localIdx>(globalIds.size()) == sub.nCells());
        for (auto gid : globalIds)
        {
            REQUIRE(gid < 16);
            globalSeen[static_cast<std::size_t>(gid)] = true;
        }
    }
    // Every global cell must appear exactly once across all partitions
    for (bool seen : globalSeen)
        REQUIRE(seen);
}

// --- Cycle 5: ghost cell data in stencilDB ---

TEST_CASE("extractSubMesh - stores ghost cell data in stencilDB")
{
    auto exec = NeoN::SerialExecutor {};
    auto mesh = NeoN::createUniform2DGrid(exec, 4, 4); // 16 cells
    auto cellPart = NeoN::partition::partitionMesh(mesh, 4);

    for (int p = 0; p < 4; ++p)
    {
        auto sub = NeoN::partition::extractSubMesh(mesh, cellPart, p);

        REQUIRE(sub.stencilDB().contains("partition::ghostCellGlobalIds"));
        REQUIRE(sub.stencilDB().contains("partition::ghostCellVolumes"));
        REQUIRE(sub.stencilDB().contains("partition::ghostCellCentres"));

        auto& ghostIds = *sub.stencilDB().get<std::shared_ptr<std::vector<NeoN::localIdx>>>(
            "partition::ghostCellGlobalIds"
        );
        auto& ghostVols = *sub.stencilDB().get<std::shared_ptr<std::vector<NeoN::scalar>>>(
            "partition::ghostCellVolumes"
        );
        auto& ghostCentres = *sub.stencilDB().get<std::shared_ptr<std::vector<NeoN::Vec3>>>(
            "partition::ghostCellCentres"
        );

        // All arrays same size
        REQUIRE(ghostIds.size() == ghostVols.size());
        REQUIRE(ghostIds.size() == ghostCentres.size());

        // Ghost cells must be from other partitions
        for (auto gid : ghostIds)
        {
            REQUIRE(cellPart[static_cast<std::size_t>(gid)] != p);
        }

        // No duplicate ghost cells
        std::set<NeoN::localIdx> uniqueGhosts(ghostIds.begin(), ghostIds.end());
        REQUIRE(uniqueGhosts.size() == ghostIds.size());

        // Must have at least one ghost cell (all parts touch at least one neighbor)
        REQUIRE(ghostIds.size() > 0);
    }
}

// --- Cycle 6: sub-mesh cell count ---

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
