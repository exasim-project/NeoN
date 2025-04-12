// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("tokenList")
{
    NeoN::TokenList tokenList;
    REQUIRE(tokenList.empty());

    tokenList.insert(NeoN::label(1));
    tokenList.insert(NeoN::scalar(2.0));
    tokenList.insert(std::string("string"));

    REQUIRE(tokenList.size() == 3);

    SECTION("GetToken")
    {
        REQUIRE(tokenList.size() == 3);
        REQUIRE(tokenList.get<NeoN::label>(0) == 1);
        REQUIRE(tokenList.get<NeoN::scalar>(1) == 2.0);
        REQUIRE(tokenList.get<std::string>(2) == "string");
    }

    SECTION("RemoveToken")
    {
        tokenList.remove(1);
        REQUIRE(tokenList.size() == 2);
        REQUIRE(tokenList.get<NeoN::label>(0) == 1);
        REQUIRE(tokenList.get<std::string>(1) == "string");
    }

    SECTION("check bad any cast")
    {
        REQUIRE_THROWS_AS(tokenList.get<std::string>(0), std::bad_any_cast);
    }

    SECTION("PopToken")
    {
        REQUIRE(tokenList.size() == 3);
        auto firstToken = tokenList.popFront<NeoN::label>();
        auto secondToken = tokenList.popFront<NeoN::scalar>();
        auto thirdToken = tokenList.popFront<std::string>();
        REQUIRE(firstToken == 1);
        REQUIRE(secondToken == 2.0);
        REQUIRE(thirdToken == "string");
        REQUIRE(tokenList.size() == 0);
    }

    SECTION("Next Token")
    {
        tokenList = NeoN::TokenList({NeoN::label(1), NeoN::scalar(2.0), std::string("string")});
        REQUIRE(tokenList.size() == 3);
        REQUIRE(tokenList.next<NeoN::label>() == 1);
        REQUIRE(tokenList.next<NeoN::scalar>() == 2.0);
        REQUIRE(tokenList.next<std::string>() == "string");
        REQUIRE_THROWS_AS(tokenList.next<NeoN::label>(), std::out_of_range);
    }

    SECTION("Access out of bound index")
    {
        REQUIRE_THROWS_AS(tokenList.get<NeoN::label>(3), std::out_of_range);
    }
}
