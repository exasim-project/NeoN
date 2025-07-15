// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>

#include "NeoN/NeoN.hpp"


bool validateCustomDoc(const NeoN::Document& doc)
{
    return doc.contains("name") && doc.contains("testValue");
}

class CustomDocument
{
public:

    CustomDocument() : doc_(NeoN::Document({{"name", ""}, {"testValue", 0}}, validateCustomDoc)) {}

    CustomDocument(const NeoN::Document& doc) : doc_(doc) {}

    CustomDocument(const std::string& name, const double& testValue)
        : doc_(NeoN::Document({{"name", name}, {"testValue", testValue}}, validateCustomDoc))
    {}

    std::string& name() { return doc_.get<std::string>("name"); }

    const std::string& name() const { return doc_.get<std::string>("name"); }

    double testValue() const { return doc_.get<double>("testValue"); }

    double& testValue() { return doc_.get<double>("testValue"); }

    NeoN::Document& doc() { return doc_; }

    const NeoN::Document& doc() const { return doc_; }

    std::string id() const { return doc_.id(); }

    static std::string typeName() { return "CustomDocument"; }

private:

    NeoN::Document doc_;
};

class CustomCollection : public NeoN::CollectionMixin<CustomDocument>
{
public:

    CustomCollection(NeoN::Database& db, std::string name)
        : NeoN::CollectionMixin<CustomDocument>(db, name)
    {}

    bool contains(const std::string& id) const { return docs_.contains(id); }

    bool insert(const CustomDocument& cc)
    {
        std::string id = cc.id();
        if (contains(id))
        {
            return false;
        }
        docs_.emplace(id, cc);
        return true;
    }

    static CustomCollection& instance(NeoN::Database& db, std::string name)
    {
        NeoN::Collection& col = db.insert(name, CustomCollection(db, name));
        return col.as<CustomCollection>();
    }
};
