// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors
#pragma once

#include <string>

#include "NeoN/core/database/database.hpp"
#include "NeoN/core/database/collection.hpp"
#include "NeoN/core/database/document.hpp"
#include "NeoN/core/database/fieldCollection.hpp"


namespace NeoN::finiteVolume::cellCentred
{

class OldTimeDocument
{
public:

    OldTimeDocument(const Document& doc);

    OldTimeDocument(
        std::string nextTime, std::string previousTime, std::string currentTime, int32_t level
    );

    std::string& nextTime();

    const std::string& nextTime() const;

    std::string& previousTime();

    const std::string& previousTime() const;

    std::string& currentTime();

    const std::string& currentTime() const;

    int32_t& level();

    const int32_t& level() const;

    Document& doc();

    const Document& doc() const;

    std::string id() const;

    static std::string typeName();

private:

    Document doc_;
};


class OldTimeCollection : public CollectionMixin<OldTimeDocument>
{
public:

    OldTimeCollection(Database& db, std::string name, std::string fieldCollectionName);

    bool contains(const std::string& id) const;

    bool insert(const OldTimeDocument& cc);

    std::string findNextTime(std::string id) const;

    std::string findPreviousTime(std::string id) const;

    OldTimeDocument& oldTimeDoc(const std::string& id);

    const OldTimeDocument& oldTimeDoc(const std::string& id) const;

    template<typename VectorType>
    VectorType& getOrInsert(std::string idOfNextVector)
    {
        std::string nextId = findNextTime(idOfNextVector);
        VectorCollection& fieldCollection = VectorCollection::instance(db(), fieldCollectionName_);

        if (nextId != "") // oldVector is already registered
        {
            OldTimeDocument& oldTimeDocument = oldTimeDoc(nextId);
            return fieldCollection.fieldDoc(oldTimeDocument.previousTime()).field<VectorType>();
        }
        VectorDocument& fieldDoc = fieldCollection.fieldDoc(idOfNextVector);

        std::string oldTimeName = fieldDoc.field<VectorType>().name + "_0";
        VectorType& oldVector =
            fieldCollection.registerVector<VectorType>(CreateFromExistingVector<VectorType> {
                .name = oldTimeName,
                .field = fieldDoc.field<VectorType>(),
                .timeIndex = fieldDoc.timeIndex() - 1,
                .iterationIndex = fieldDoc.iterationIndex(),
                .subCycleIndex = fieldDoc.subCycleIndex()
            });
        OldTimeDocument oldTimeDocument(fieldDoc.field<VectorType>().key, oldVector.key, "", -1);
        setCurrentVectorAndLevel(oldTimeDocument);
        insert(oldTimeDocument);
        return oldVector;
    }

    template<typename VectorType>
    const VectorType& get(std::string idOfNextVector) const
    {
        std::string nextId = findNextTime(idOfNextVector);
        const VectorCollection& fieldCollection =
            VectorCollection::instance(db(), fieldCollectionName_);

        if (nextId != "") // oldVector has to be registered
        {
            const OldTimeDocument& oldTimeDocument = oldTimeDoc(nextId);
            return fieldCollection.fieldDoc(oldTimeDocument.previousTime()).field<VectorType>();
        }
        else
        {
            // TODO replace with NF_THROW
            NF_ERROR_EXIT("Old field not found");
        }
    }

    static OldTimeCollection&
    instance(Database& db, std::string name, std::string fieldCollectionName);

    static const OldTimeCollection& instance(const Database& db, std::string name);

    static OldTimeCollection& instance(VectorCollection& fieldCollection);

    static const OldTimeCollection& instance(const VectorCollection& fieldCollection);

private:

    /** */
    void setCurrentVectorAndLevel(OldTimeDocument& oldTimeDoc);

    std::string fieldCollectionName_;
};

/**
 * @brief Retrieves the old time field of a given field.
 *
 * This function retrieves the old time field of a given field
 *
 * @param field The field to retrieve the old time field from.
 * @return The old time field.
 */
template<typename VectorType>
VectorType& oldTime(VectorType& field)
{
    VectorCollection& fieldCollection = VectorCollection::instance(field);
    OldTimeCollection& oldTimeCollection = OldTimeCollection::instance(fieldCollection);
    return oldTimeCollection.getOrInsert<VectorType>(field.key);
}

/**
 * @brief Retrieves the old time field of a given field (const version).
 *
 * This function retrieves the old time field of a given field
 *
 * @param field The field to retrieve the old time field from.
 * @return The old time field.
 */
template<typename VectorType>
const VectorType& oldTime(const VectorType& field)
{
    const VectorCollection& fieldCollection = VectorCollection::instance(field);
    const OldTimeCollection& oldTimeCollection = OldTimeCollection::instance(fieldCollection);
    return oldTimeCollection.get<VectorType>(field.key);
}

} // namespace NeoN
