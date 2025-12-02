// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/database/fieldCollection.hpp"

namespace NeoN::finiteVolume::cellCentred
{

// Initialize the static member
bool validateVectorDoc(const Document& doc)
{
    return doc.contains("name") && doc.contains("timeIndex") && doc.contains("iterationIndex")
        && doc.contains("subCycleIndex") && hasId(doc) && doc.contains("field");
}

VectorDocument::VectorDocument(const Document& doc) : doc_(doc, validateVectorDoc) {}

std::string VectorDocument::id() const { return doc_.id(); }

std::string VectorDocument::typeName() { return "VectorDocument"; }

Document& VectorDocument::doc() { return doc_; }

const Document& VectorDocument::doc() const { return doc_; }

std::string VectorDocument::name() const { return doc_.get<std::string>("name"); }

std::string& VectorDocument::name() { return doc_.get<std::string>("name"); }

std::int64_t VectorDocument::timeIndex() const { return doc_.get<std::int64_t>("timeIndex"); }

std::int64_t& VectorDocument::timeIndex() { return doc_.get<std::int64_t>("timeIndex"); }

std::int64_t VectorDocument::iterationIndex() const
{
    return doc_.get<std::int64_t>("iterationIndex");
}

std::int64_t& VectorDocument::iterationIndex() { return doc_.get<std::int64_t>("iterationIndex"); }

std::int64_t VectorDocument::subCycleIndex() const
{
    return doc_.get<std::int64_t>("subCycleIndex");
}

std::int64_t& VectorDocument::subCycleIndex() { return doc_.get<std::int64_t>("subCycleIndex"); }

VectorCollection::VectorCollection(NeoN::Database& db, std::string name)
    : NeoN::CollectionMixin<VectorDocument>(db, name)
{}


bool VectorCollection::contains(const std::string& id) const { return docs_.contains(id); }

std::string VectorCollection::insert(const VectorDocument& cc)
{
    std::string id = cc.id();
    if (contains(id))
    {
        return "";
    }
    docs_.emplace(id, cc);
    return id;
}

VectorDocument& VectorCollection::fieldDoc(const std::string& id) { return docs_.at(id); }

const VectorDocument& VectorCollection::fieldDoc(const std::string& id) const
{
    return docs_.at(id);
}

VectorCollection& VectorCollection::instance(NeoN::Database& db, std::string name)
{
    NeoN::Collection& col = db.insert(name, VectorCollection(db, name));
    return col.as<VectorCollection>();
}

const VectorCollection& VectorCollection::instance(const NeoN::Database& db, std::string name)
{
    const NeoN::Collection& col = db.at(name);
    return col.as<VectorCollection>();
}

} // namespace NeoN::finiteVolume::cellCentred
