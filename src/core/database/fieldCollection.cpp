// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#include "NeoN/core/database/fieldCollection.hpp"

namespace NeoN::finiteVolume::cellCentred
{

// Initialize the static member
bool validateFieldDoc(const Document& doc)
{
    return doc.contains("name") && doc.contains("timeIndex") && doc.contains("iterationIndex")
        && doc.contains("subCycleIndex") && hasId(doc) && doc.contains("field");
}

FieldDocument::FieldDocument(const Document& doc) : doc_(doc, validateFieldDoc) {}

std::string FieldDocument::id() const { return doc_.id(); }

std::string FieldDocument::typeName() { return "FieldDocument"; }

Document& FieldDocument::doc() { return doc_; }

const Document& FieldDocument::doc() const { return doc_; }

std::string FieldDocument::name() const { return doc_.get<std::string>("name"); }

std::string& FieldDocument::name() { return doc_.get<std::string>("name"); }

std::int64_t FieldDocument::timeIndex() const { return doc_.get<std::int64_t>("timeIndex"); }

std::int64_t& FieldDocument::timeIndex() { return doc_.get<std::int64_t>("timeIndex"); }

std::int64_t FieldDocument::iterationIndex() const
{
    return doc_.get<std::int64_t>("iterationIndex");
}

std::int64_t& FieldDocument::iterationIndex() { return doc_.get<std::int64_t>("iterationIndex"); }

std::int64_t FieldDocument::subCycleIndex() const
{
    return doc_.get<std::int64_t>("subCycleIndex");
}

std::int64_t& FieldDocument::subCycleIndex() { return doc_.get<std::int64_t>("subCycleIndex"); }


FieldCollection::FieldCollection(NeoN::Database& db, std::string name)
    : NeoN::CollectionMixin<FieldDocument>(db, name)
{}

bool FieldCollection::contains(const std::string& id) const { return docs_.contains(id); }

std::string FieldCollection::insert(const FieldDocument& cc)
{
    std::string id = cc.id();
    if (contains(id))
    {
        return "";
    }
    docs_.emplace(id, cc);
    return id;
}

FieldDocument& FieldCollection::fieldDoc(const std::string& id) { return docs_.at(id); }

const FieldDocument& FieldCollection::fieldDoc(const std::string& id) const { return docs_.at(id); }

FieldCollection& FieldCollection::instance(NeoN::Database& db, std::string name)
{
    NeoN::Collection& col = db.insert(name, FieldCollection(db, name));
    return col.as<FieldCollection>();
}

const FieldCollection& FieldCollection::instance(const NeoN::Database& db, std::string name)
{
    const NeoN::Collection& col = db.at(name);
    return col.as<FieldCollection>();
}

} // namespace NeoN::finiteVolume::cellCentred
