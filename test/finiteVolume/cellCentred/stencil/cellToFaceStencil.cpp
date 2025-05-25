// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 FoamAdapter authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

TEST_CASE("cell To Face Stencil")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto mesh = NeoN::create1DUniformMesh(exec, nCells);

    SECTION("cellToFaceStencil_" + execName)
    {
        fvcc::CellToFaceStencil cellToFaceStencil(mesh);
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> stencil =
            cellToFaceStencil.computeStencil();

        // auto hostStencil = stencil.copyToHost();
        // auto stencilView = hostStencil.view();
        //
        // for (auto celli = 0; celli < mesh.cells().size(); celli++)
        // {
        //     std::unordered_set<Foam::label> faceSet;
        //     REQUIRE(stencilView.view(celli).size() == mesh.cells()[celli].size());
        //     for (auto facei : mesh.cells()[celli])
        //     {
        //         faceSet.insert(facei);
        //     }
        //
        //     for (auto facei : stencilView.view(celli))
        //     {
        //         REQUIRE(faceSet.contains(facei));
        //     }
        // }
    }
}
