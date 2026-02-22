// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/catch_approx.hpp>

#include <random>

#include "NeoN/NeoN.hpp"

struct ApproxScalar
{
    NeoN::scalar margin;
    bool operator()(double rhs, double lhs) const
    {
        return Catch::Approx(rhs).margin(margin) == lhs;
    }
};

/* comparison function for volumeFields */
template<typename FieldType, typename Compare>
void compare(const FieldType& a, const FieldType& b, Compare comp)
{
    auto aHost = a.copyToHost();
    auto bHost = b.copyToHost();
    REQUIRE(aHost.size() == bHost.size());
    REQUIRE_THAT(aHost.view(), Catch::Matchers::RangeEquals(bHost.view(), comp));
}
