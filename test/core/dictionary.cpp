// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/core/dictionary.hpp"

TEST_CASE("Dictionary operations", "[dictionary]")
{
    NeoN::Dictionary dict;

    SECTION("Insert and retrieve values")
    {
        dict.insert("key1", 42);
        dict.insert("key2", std::string("Hello"));

        REQUIRE(std::any_cast<int>(dict["key1"]) == 42);
        REQUIRE(dict.getRef<int>("key1") == 42);
        REQUIRE(std::any_cast<std::string>(dict["key2"]) == "Hello");
        REQUIRE(dict.getRef<std::string>("key2") == "Hello");
    }

    SECTION("check values")
    {
        dict.insert("key", 42);
        dict["key"] = 43;

        REQUIRE(dict.contains("key"));
        REQUIRE(!dict.isDict("key"));
    }

    SECTION("Modify values")
    {
        dict.insert("key", 42);
        dict["key"] = 43;

        REQUIRE(dict.getVal<int>("key") == 43);
    }

    SECTION("remove values")
    {
        dict.insert("key", 42);
        dict["key"] = 43;
        dict.remove("key");

        REQUIRE(!dict.contains("key"));
    }

    SECTION("Access non-existent key")
    {
        REQUIRE_THROWS_AS(dict["non_existent_key"], std::out_of_range);
        REQUIRE_THROWS_AS(dict.getVal<int>("non_existent_key"), std::out_of_range);
    }

    SECTION("subDict")
    {
        NeoN::Dictionary subDict;
        subDict.insert("key1", 42);
        subDict.insert("key2", std::string("Hello"));

        dict.insert("subDict", subDict);

        NeoN::Dictionary& sDict = dict.subDict("subDict");
        REQUIRE(sDict.getVal<int>("key1") == 42);
        REQUIRE(sDict.getVal<std::string>("key2") == "Hello");

        sDict.getRef<int>("key1") = 100;

        // check if the value is modified
        REQUIRE(dict.isDict("subDict"));
        NeoN::Dictionary& sDict2 = dict.subDict("subDict");
        REQUIRE(sDict2.getVal<int>("key1") == 100);
    }

    SECTION("initialize with map")
    {
        NeoN::Dictionary dictInit({{"key1", 42}, {"key2", std::string("Hello")}});

        REQUIRE(dictInit.getVal<int>("key1") == 42);
        REQUIRE(dictInit.getVal<std::string>("key2") == "Hello");
    }
}
