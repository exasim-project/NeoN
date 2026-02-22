// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/linearAlgebra/blockVector.hpp"

TEST_CASE("BlockVector")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Construction and access " + execName)
    {
        NeoN::la::BlockVector bv(exec, 2, 1, 0.0);

        REQUIRE(bv.nBlocks() == 2);
        REQUIRE(bv.nCells() == 1);
        REQUIRE(bv.totalSize() == 2);
        REQUIRE(bv.vector().size() == 2);
    }

    SECTION("BlockVectorView operator() access " + execName)
    {
        NeoN::la::BlockVector bv(exec, 2, 1, 0.0);
        NeoN::Vector<NeoN::scalar> srcA(exec, std::vector<NeoN::scalar> {2.0});
        NeoN::Vector<NeoN::scalar> srcB(exec, std::vector<NeoN::scalar> {3.0});
        bv.copyBlockFrom(0, srcA);
        bv.copyBlockFrom(1, srcB);

        auto bvView = bv.view();
        NeoN::Vector<NeoN::scalar> result(exec, 2, 0.0);
        auto rv = result.view();

        NeoN::parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const NeoN::localIdx) {
                rv[0] = bvView(0)[0];
                rv[1] = bvView(1)[0];
            },
            "BlockVectorView_access"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 2.0);
        REQUIRE(hostResult.view()[1] == 3.0);
    }

    SECTION("Scatter and gather " + execName)
    {
        NeoN::la::BlockVector bv(exec, 2, 1, 0.0);

        NeoN::Vector<NeoN::scalar> fieldA(exec, std::vector<NeoN::scalar> {2.0});
        NeoN::Vector<NeoN::scalar> fieldB(exec, std::vector<NeoN::scalar> {3.0});
        bv.copyBlockFrom(0, fieldA);
        bv.copyBlockFrom(1, fieldB);

        auto hostData = bv.vector().copyToHost();
        REQUIRE(hostData.view()[0] == 2.0);
        REQUIRE(hostData.view()[1] == 3.0);

        NeoN::Vector<NeoN::scalar> outA(exec, 1, 0.0);
        NeoN::Vector<NeoN::scalar> outB(exec, 1, 0.0);
        bv.copyBlockTo(0, outA);
        bv.copyBlockTo(1, outB);

        auto hostA = outA.copyToHost();
        auto hostB = outB.copyToHost();
        REQUIRE(hostA.view()[0] == 2.0);
        REQUIRE(hostB.view()[0] == 3.0);
    }

    SECTION("Multi-cell scatter and gather " + execName)
    {
        NeoN::la::BlockVector bv(exec, 2, 3, 0.0);

        NeoN::Vector<NeoN::scalar> fieldA(exec, std::vector<NeoN::scalar> {1.0, 2.0, 3.0});
        NeoN::Vector<NeoN::scalar> fieldB(exec, std::vector<NeoN::scalar> {4.0, 5.0, 6.0});
        bv.copyBlockFrom(0, fieldA);
        bv.copyBlockFrom(1, fieldB);

        auto hostData = bv.vector().copyToHost();
        REQUIRE(hostData.view()[0] == 1.0);
        REQUIRE(hostData.view()[1] == 2.0);
        REQUIRE(hostData.view()[2] == 3.0);
        REQUIRE(hostData.view()[3] == 4.0);
        REQUIRE(hostData.view()[4] == 5.0);
        REQUIRE(hostData.view()[5] == 6.0);

        NeoN::Vector<NeoN::scalar> outA(exec, 3, 0.0);
        NeoN::Vector<NeoN::scalar> outB(exec, 3, 0.0);
        bv.copyBlockTo(0, outA);
        bv.copyBlockTo(1, outB);

        auto hostA = outA.copyToHost();
        auto hostB = outB.copyToHost();
        auto hostFieldA = fieldA.copyToHost();
        auto hostFieldB = fieldB.copyToHost();
        for (NeoN::localIdx i = 0; i < 3; ++i)
        {
            REQUIRE(hostA.view()[i] == hostFieldA.view()[i]);
            REQUIRE(hostB.view()[i] == hostFieldB.view()[i]);
        }
    }
}
