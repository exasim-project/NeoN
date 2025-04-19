// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include <Kokkos_Core.hpp>

#include "NeoN/NeoN.hpp"


TEST_CASE("parallelFor")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Vector<NeoN::scalar> field(exec, 5);
    const NeoN::Vector<NeoN::scalar> field1(exec, 4, 3.0);
    NeoN::fill(field, 2.0);

    auto fieldStdView = field.view();
    auto fieldNFView = NeoN::View(fieldStdView);

    NeoN::parallelFor(
        exec, {0, 5}, KOKKOS_LAMBDA(const NeoN::localIdx i) { fieldNFView[i] *= 2.0; }
    );
    REQUIRE(fieldNFView.failureIndex == 0);

#ifdef NF_DEBUGC
// TODO: on MSCV this results in a non terminating loop
// so for now we deactivate it on MSVC since it a debugging helper
#ifndef _MSC_VER
    fieldNFView.abort = false;
    NeoN::parallelFor(
        exec, {5, 6}, KOKKOS_LAMBDA(const localIdx i) { fieldNFView[i] *= 2.0; }
    );
    REQUIRE(fieldNFView.failureIndex == 5);
#endif
#endif

    auto fieldHost = field.copyToHost();
    auto fieldNFViewHost = fieldHost.view();

#ifdef NF_DEBUG
// TODO: on MSCV this results in a non terminating loop
// so for now we deactivate it on MSVC since it a debugging helper
#ifndef _MSC_VER
    fieldNFViewHost.abort = false;
    SECTION("detects out of range")
    {
        [[maybe_unused]] auto tmp = fieldNFViewHost[5];
        REQUIRE(fieldNFViewHost.failureIndex == 5);
    }
#endif
#endif

    // some checking if everything is correct
    SECTION("can access elements")
    {
        REQUIRE(fieldNFViewHost[0] == 4.0);
        REQUIRE(fieldNFViewHost[1] == 4.0);
        REQUIRE(fieldNFViewHost[2] == 4.0);
        REQUIRE(fieldNFViewHost[3] == 4.0);
        REQUIRE(fieldNFViewHost[4] == 4.0);
    }

    // unpack multiple views, and check correctness.
    SECTION("views")
    {
        auto fieldHost = field.copyToHost();
        auto field1Host = field1.copyToHost();

        auto [viewHost, view1Host] = views(fieldHost, field1Host);

        REQUIRE(viewHost.size() == 5);
        REQUIRE(viewHost[0] == 4.0);
        REQUIRE(viewHost[1] == 4.0);
        REQUIRE(viewHost[2] == 4.0);
        REQUIRE(viewHost[3] == 4.0);
        REQUIRE(viewHost[4] == 4.0);

        REQUIRE(view1Host.size() == 4);
        REQUIRE(view1Host[0] == 3.0);
        REQUIRE(view1Host[1] == 3.0);
        REQUIRE(view1Host[2] == 3.0);
        REQUIRE(view1Host[3] == 3.0);
    }
};
