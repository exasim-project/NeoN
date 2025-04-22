// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

#include <Kokkos_Core.hpp>

TEST_CASE("segmentedVector")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Constructor from sizes " + execName)
    {
        NeoN::SegmentedVector<NeoN::label, NeoN::localIdx> segVector(exec, 10, 5);
        auto [values, segments] = segVector.views();

        REQUIRE(values.size() == 10);
        REQUIRE(segments.size() == 6);

        REQUIRE(segVector.numSegments() == 5);
        REQUIRE(segVector.size() == 10);
    }

    SECTION("Constructor from field " + execName)
    {
        NeoN::Vector<NeoN::label> values(exec, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        NeoN::Vector<NeoN::localIdx> segments(exec, {0, 2, 4, 6, 8, 10});

        NeoN::SegmentedVector<NeoN::label, NeoN::localIdx> segVector(values, segments);

        REQUIRE(segVector.values().size() == 10);
        REQUIRE(segVector.segments().size() == 6);
        REQUIRE(segVector.numSegments() == 5);

        REQUIRE(segVector.values().exec() == exec);

        auto hostValues = segVector.values().copyToHost();
        auto hostSegments = segVector.segments().copyToHost();

        REQUIRE(hostValues.view()[5] == 5);
        REQUIRE(hostSegments.view()[2] == 4);

        SECTION("loop over segments")
        {
            auto [valueView, segment] = segVector.views();
            auto segView = segVector.view();
            NeoN::Vector<NeoN::label> result(exec, 5);

            NeoN::fill(result, 0);
            auto resultView = result.view();

            parallelFor(
                exec,
                {0, segVector.numSegments()},
                KOKKOS_LAMBDA(const NeoN::localIdx segI) {
                    // check if it works with bounds
                    auto [bStart, bEnd] = segView.bounds(segI);
                    auto bVals = valueView.subview(bStart, bEnd - bStart);
                    for (auto& val : bVals)
                    {
                        resultView[segI] += val;
                    }

                    // check if it works with range
                    auto [rStart, rLength] = segView.range(segI);
                    auto rVals = valueView.subview(rStart, rLength);
                    for (auto& val : rVals)
                    {
                        resultView[segI] += val;
                    }

                    // check with subview
                    auto vals = segView.view(segI);
                    for (auto& val : vals)
                    {
                        resultView[segI] += val;
                    }
                }
            );

            auto hostResult = result.copyToHost();
            REQUIRE(hostResult.view()[0] == 1 * 3);
            REQUIRE(hostResult.view()[1] == 5 * 3);
            REQUIRE(hostResult.view()[2] == 9 * 3);
            REQUIRE(hostResult.view()[3] == 13 * 3);
            REQUIRE(hostResult.view()[4] == 17 * 3);
        }
    }

    SECTION("Constructor from list with offsets " + execName)
    {
        NeoN::Vector<NeoN::localIdx> offsets(exec, {1, 2, 3, 4, 5});
        NeoN::SegmentedVector<NeoN::label, NeoN::localIdx> segVector(offsets);

        auto hostSegments = segVector.segments().copyToHost();
        REQUIRE(hostSegments.view()[0] == 0);
        REQUIRE(hostSegments.view()[1] == 1);
        REQUIRE(hostSegments.view()[2] == 3);
        REQUIRE(hostSegments.view()[3] == 6);
        REQUIRE(hostSegments.view()[4] == 10);
        REQUIRE(hostSegments.view()[5] == 15);

        auto hostOffsets = offsets.copyToHost();
        REQUIRE(hostOffsets.view()[0] == 1);
        REQUIRE(hostOffsets.view()[1] == 2);
        REQUIRE(hostOffsets.view()[2] == 3);
        REQUIRE(hostOffsets.view()[3] == 4);
        REQUIRE(hostOffsets.view()[4] == 5);

        REQUIRE(segVector.size() == 15);

        SECTION("update values")
        {
            auto segView = segVector.view();
            NeoN::Vector<NeoN::label> result(exec, 5);

            NeoN::fill(result, 0);
            auto resultView = result.view();


            parallelFor(
                exec,
                {0, segVector.numSegments()},
                KOKKOS_LAMBDA(const NeoN::localIdx segI) {
                    // fill values
                    auto vals = segView.view(segI);
                    for (auto& val : vals)
                    {
                        val = segI;
                    }

                    // accumulate values
                    for (const auto& val : vals)
                    {
                        resultView[segI] += val;
                    }
                }
            );

            auto hostResult = result.copyToHost();
            REQUIRE(hostResult.view()[0] == 0 * 1);
            REQUIRE(hostResult.view()[1] == 1 * 2);
            REQUIRE(hostResult.view()[2] == 2 * 3);
            REQUIRE(hostResult.view()[3] == 3 * 4);
            REQUIRE(hostResult.view()[4] == 4 * 5);
        }
    }
}
