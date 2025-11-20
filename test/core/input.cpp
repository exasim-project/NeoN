// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/core/input.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"


struct TestInput
{

    static TestInput read(const NeoN::Dictionary& dict)
    {
        TestInput ti;
        ti.label_ = dict.getVal<NeoN::label>("label");
        ti.scalar_ = dict.getVal<NeoN::scalar>("scalar");
        ti.string_ = dict.getVal<std::string>("string");
        return ti;
    }

    static TestInput read(const NeoN::TokenList& tl)
    {
        TestInput ti;
        ti.label_ = tl.get<NeoN::label>(0);
        ti.scalar_ = tl.get<NeoN::scalar>(1);
        ti.string_ = tl.get<std::string>(2);
        return ti;
    }
    NeoN::label label_;
    NeoN::scalar scalar_;
    std::string string_;
};

TEST_CASE("Input")
{
    SECTION("read from TokenList")
    {
        NeoN::TokenList tokenList;
        REQUIRE(tokenList.empty());

        tokenList.insert(1);
        tokenList.insert(2.0);
        tokenList.insert(std::string("string"));

        REQUIRE(tokenList.size() == 3);

        NeoN::Input input = tokenList;

        TestInput ti = NeoN::read<TestInput>(input);
        REQUIRE(ti.label_ == 1);
        REQUIRE(ti.scalar_ == 2.0);
        REQUIRE(ti.string_ == "string");
    }

    SECTION("read from Dictionary")
    {
        NeoN::Dictionary dict;

        dict.insert("label", 1);
        dict.insert("scalar", 2.0);
        dict.insert("string", std::string("string"));

        REQUIRE(dict.keys().size() == 3);

        NeoN::Input input = dict;

        TestInput ti = NeoN::read<TestInput>(input);
        REQUIRE(ti.label_ == 1);
        REQUIRE(ti.scalar_ == 2.0);
        REQUIRE(ti.string_ == "string");
    }
}
