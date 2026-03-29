// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "executorGenerator.hpp"


#define SECTION_IF(COND, ...)                                                                      \
    if (COND) SECTION(__VA_ARGS__)

struct ApproxScalar
{
    NeoN::scalar margin;
    bool operator()(double rhs, double lhs) const
    {
        return Catch::Approx(rhs).margin(margin) == lhs;
    }
};

struct EqualInt
{
    bool operator()(auto rhs, auto lhs) const { return rhs == lhs; }
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
