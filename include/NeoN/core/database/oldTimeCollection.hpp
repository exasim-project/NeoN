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

    template<typename FieldType>
    FieldType& getOrInsert(std::string idOfNextField)
    {
        std::string nextId = findNextTime(idOfNextField);
        FieldCollection& fieldCollection = FieldCollection::instance(db(), fieldCollectionName_);

        if (nextId != "") // oldField is already registered
        {
            OldTimeDocument& oldTimeDocument = oldTimeDoc(nextId);
            return fieldCollection.fieldDoc(oldTimeDocument.previousTime()).field<FieldType>();
        }
        FieldDocument& fieldDoc = fieldCollection.fieldDoc(idOfNextField);

        std::string oldTimeName = fieldDoc.field<FieldType>().name + "_0";
        FieldType& oldField =
            fieldCollection.registerField<FieldType>(CreateFromExistingField<FieldType> {
                .name = oldTimeName,
                .field = fieldDoc.field<FieldType>(),
                .timeIndex = fieldDoc.timeIndex() - 1,
                .iterationIndex = fieldDoc.iterationIndex(),
                .subCycleIndex = fieldDoc.subCycleIndex()
            });
        OldTimeDocument oldTimeDocument(fieldDoc.field<FieldType>().key, oldField.key, "", -1);
        setCurrentFieldAndLevel(oldTimeDocument);
        insert(oldTimeDocument);
        return oldField;
    }

    template<typename FieldType>
    const FieldType& get(std::string idOfNextField) const
    {
        std::string nextId = findNextTime(idOfNextField);
        const FieldCollection& fieldCollection =
            FieldCollection::instance(db(), fieldCollectionName_);

        if (nextId != "") // oldField has to be registered
        {
            const OldTimeDocument& oldTimeDocument = oldTimeDoc(nextId);
            return fieldCollection.fieldDoc(oldTimeDocument.previousTime()).field<FieldType>();
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

    static OldTimeCollection& instance(FieldCollection& fieldCollection);

    static const OldTimeCollection& instance(const FieldCollection& fieldCollection);

private:

    /** */
    void setCurrentFieldAndLevel(OldTimeDocument& oldTimeDoc);

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
template<typename FieldType>
FieldType& oldTime(FieldType& field)
{
    FieldCollection& fieldCollection = FieldCollection::instance(field);
    OldTimeCollection& oldTimeCollection = OldTimeCollection::instance(fieldCollection);
    return oldTimeCollection.getOrInsert<FieldType>(field.key);
}

/**
 * @brief Retrieves the old time field of a given field (const version).
 *
 * This function retrieves the old time field of a given field
 *
 * @param field The field to retrieve the old time field from.
 * @return The old time field.
 */
template<typename FieldType>
const FieldType& oldTime(const FieldType& field)
{
    const FieldCollection& fieldCollection = FieldCollection::instance(field);
    const OldTimeCollection& oldTimeCollection = OldTimeCollection::instance(fieldCollection);
    return oldTimeCollection.get<FieldType>(field.key);
}

} // namespace NeoN
