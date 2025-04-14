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
    auto fieldNFViewHost = NeoN::View(fieldHost.view());

#ifdef NF_DEBUG
// TODO: on MSCV this results in a non terminating loop
// so for now we deactivate it on MSVC since it a debugging helper
#ifndef _MSC_VER
    fieldNFViewHost.abort = false;
    SECTION("detects out of range")
    {
        auto tmp = fieldNFViewHost[5];
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
};
