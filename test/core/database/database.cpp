// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoN/NeoN.hpp"

#include "customTestCollection.hpp"

CustomDocument createDoc(std::string name, int value)
{
    return CustomDocument(NeoN::Document({{"name", name}, {"testValue", value}}));
}

TEST_CASE("Database")
{
    NeoN::Database db;

    SECTION("insert")
    {
        db.insert("collection1", CustomCollection(db, "collection1"));
        db.insert("collection2", CustomCollection(db, "collection2"));
        REQUIRE(db.contains("collection1"));
        REQUIRE(db.contains("collection2"));
        REQUIRE_FALSE(db.contains("collection3"));
        REQUIRE(db.size() == 2);
    }

    SECTION("get collections")
    {
        auto& collection1 = db.insert("collection1", CustomCollection(db, "collection1"));

        auto& collection2 = db.at("collection1");

        REQUIRE(&collection1 == &collection2);

        SECTION("check database access from collection") { REQUIRE(&db == &collection1.db()); }

        REQUIRE(collection1.size() == 0);

        SECTION("get non existing collection") { REQUIRE_THROWS(db.at("doesNotExist")); }
    }

    SECTION("erase")
    {
        db.insert("collection1", CustomCollection(db, "collection1"));
        db.insert("collection2", CustomCollection(db, "collection2"));

        REQUIRE(db.size() == 2);

        REQUIRE(db.remove("collection1"));
        REQUIRE_FALSE(db.contains("collection1"));
        REQUIRE(db.size() == 1);

        REQUIRE_FALSE(db.remove("collection1"));
        REQUIRE(db.size() == 1);
    }


    SECTION("query")
    {
        db.insert("collection1", CustomCollection(db, "collection1"));
        db.insert("collection2", CustomCollection(db, "collection2"));

        REQUIRE(db.at<CustomCollection>("collection1").size() == 0);
        REQUIRE(db.at<CustomCollection>("collection2").size() == 0);

        auto& collection1 = db.at<CustomCollection>("collection1");
        auto& collection2 = db.at<CustomCollection>("collection2");

        collection1.insert(createDoc("doc1", 42));
        collection1.insert(createDoc("doc2", 42));
        collection2.insert(createDoc("doc3", 42));

        REQUIRE(collection1.size() == 2);
        REQUIRE(collection2.size() == 1);

        REQUIRE(collection1.name() == "collection1");
        REQUIRE(collection2.name() == "collection2");

        SECTION("find by name")
        {

            std::vector<std::string> foundKeys =
                collection1.find([](const NeoN::Document& doc)
                                 { return doc.getVal<std::string>("name") == "doc1"; });

            REQUIRE(foundKeys.size() == 1);
            REQUIRE(foundKeys[0].substr(0, 4) == "doc_");
            std::vector<std::string> keys = collection1.sortedKeys();
            REQUIRE(keys == std::vector<std::string> {"doc_0", "doc_1"});

            REQUIRE_NOTHROW(collection1.doc(foundKeys[0]));

            std::vector<std::string> foundKeys2 =
                collection1.find([](const NeoN::Document& doc)
                                 { return doc.getVal<int>("testValue") == 42; });

            REQUIRE(foundKeys2.size() == 2);
        }
    }
}
