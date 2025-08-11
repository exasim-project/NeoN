// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include <unordered_set>
#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

TEST_CASE("cell To Face Stencil")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 5;
    auto mesh = NeoN::create1DUniformMesh(exec, nCells);

    SECTION("cellToFaceStencil_" + execName)
    {
        auto cellToFaceStencil = fvcc::CellToFaceStencil(mesh);
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> stencil =
            cellToFaceStencil.computeStencil();

        auto hostStencil = stencil.copyToHost();
        auto stencilView = hostStencil.view();

        std::vector<std::vector<NeoN::localIdx>> faceExp {
            {0, 4}, // cell 0
            {0, 1}, // cell 1
            {1, 2}, // cell 2
            {2, 3}, // cell 3
            {3, 5}  // cell 4
        };

        for (auto celli = 0; celli < mesh.nCells(); celli++)
        {
            // every cell in the 1D mesh has 2 faces
            REQUIRE(stencilView.view(celli).size() == 2);
            for (auto facei = 0; facei < stencilView.view(celli).size(); facei++)
            {
                REQUIRE(stencilView.view(celli)[facei] == faceExp[celli][facei]);
            }
        }
    }
}
