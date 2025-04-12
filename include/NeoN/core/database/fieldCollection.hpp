// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <limits>
#include <string>
#include <functional>

#include "NeoFOAM/core/demangle.hpp"
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/fields/domainField.hpp"
#include "NeoFOAM/core/database/database.hpp"
#include "NeoFOAM/core/database/collection.hpp"
#include "NeoFOAM/core/database/document.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{
/**
 * @brief Validates a FieldDocument.
 *
 * This function validates a FieldDocument by checking if it contains the required fields.
 *
 * @param doc The Document to validate.
 * @return true if the Document is valid, false otherwise.
 */
bool validateFieldDoc(const Document& doc);

/**
 * @class FieldDocument
 * @brief A class representing a field document in a database.
 *
 * The FieldDocument class represents a field document in a database. It is a subclass of the
 * Document class and provides additional functionality for accessing field-specific data.
 */
class FieldDocument
{
public:

    /**
     * @brief Constructs a FieldDocument with the given field and metadata.
     *
     * @tparam FieldType The type of the field.
     * @param field The field to store in the document.
     * @param timeIndex The time index of the field.
     * @param iterationIndex The iteration index of the field.
     * @param subCycleIndex The sub-cycle index of the field.
     */
    template<class FieldType>
    FieldDocument(
        const FieldType& field,
        std::int64_t timeIndex,
        std::int64_t iterationIndex,
        std::int64_t subCycleIndex
    )
        : doc_(
            Document(
                {{"name", field.name},
                 {"timeIndex", timeIndex},
                 {"iterationIndex", iterationIndex},
                 {"subCycleIndex", subCycleIndex},
                 {"field", field}}
            ),
            validateFieldDoc
        )
    {}

    /**
     * @brief Constructs a FieldDocument with the given Document.
     *
     * @param doc The Document to construct the FieldDocument from.
     */
    FieldDocument(const Document& doc);

    /**
     * @brief Retrieves the underlying Document.
     *
     * @return Document& A reference to the underlying Document.
     */
    Document& doc();

    /**
     * @brief Retrieves the underlying Document (const version).
     *
     * @return const Document& A const reference to the underlying Document.
     */
    const Document& doc() const;


    /**
     * @brief Retrieves the unique identifier of the field collection.
     *
     * @return A string representing the unique identifier.
     */
    std::string id() const;

    /**
     * @brief Retrieves the type name of the field.
     *
     * @return A string representing the type name.
     */
    static std::string typeName();

    /**
     * @brief Retrieves the field from the document.
     *
     * @tparam FieldType The type of the field.
     * @return A reference to the field.
     */
    template<class FieldType>
    FieldType& field()
    {
        return doc_.get<FieldType&>("field");
    }

    /**
     * @brief Retrieves the field from the document (const version).
     *
     * @tparam FieldType The type of the field.
     * @return A const reference to the field.
     */
    template<class FieldType>
    const FieldType& field() const
    {
        return doc_.get<const FieldType&>("field");
    }

    /**
     * @brief Retrieves the name of the field.
     *
     * @return A string representing the name of the field.
     */
    std::string name() const;

    /**
     * @brief Retrieves the time index of the field.
     *
     * @return An integer representing the time index.
     */
    std::string& name();

    /**
     * @brief Retrieves the time index of the field.
     *
     * @return An integer representing the time index.
     */
    std::int64_t timeIndex() const;

    /**
     * @brief Retrieves the time index of the field.
     *
     * @return An integer representing the time index.
     */
    std::int64_t& timeIndex();

    /**
     * @brief Retrieves the iteration index of the field.
     *
     * @return An integer representing the iteration index.
     */
    std::int64_t iterationIndex() const;

    /**
     * @brief Retrieves the iteration index of the field.
     *
     * @return An integer representing the iteration index.
     */
    std::int64_t& iterationIndex();

    /**
     * @brief Retrieves the sub-cycle index of the field.
     *
     * @return An integer representing the sub-cycle index.
     */
    std::int64_t subCycleIndex() const;

    /**
     * @brief Retrieves the sub-cycle index of the field.
     *
     * @return An integer representing the sub-cycle index.
     */
    std::int64_t& subCycleIndex();

private:

    Document doc_; /**< The underlying Document. */
};

/**
 * @brief A function type for creating a FieldDocument.
 *
 * This function type is used to create a FieldDocument and creates a
 * registered FieldType
 *
 * @param db The database to create the FieldDocument in.
 * @return The created FieldDocument.
 */
using CreateFunction = std::function<FieldDocument(NeoFOAM::Database& db)>;

/**
 * @class FieldCollection
 * @brief A class representing a collection of field documents in a database.
 *
 * The FieldCollection class represents a collection of field documents in a database and provides
 * additional functionality for accessing field-specific data.
 */
class FieldCollection : public CollectionMixin<FieldDocument>
{
public:

    /**
     * @brief Constructs a FieldCollection with the given database and name.
     *
     * @param db The database to create the collection in.
     * @param name The name of the collection.
     */
    FieldCollection(NeoFOAM::Database& db, std::string name);

    /**
     * @brief A FieldCollection is not copyable, but moveable
     */
    FieldCollection(const FieldCollection&) = delete;

    /**
     * @brief A FieldCollection is not copyable, but moveable
     */
    FieldCollection& operator=(const FieldCollection&) = delete;

    /**
     * @brief A FieldCollection is move constructable, but not copyable
     */
    FieldCollection(FieldCollection&&) = default;

    /**
     * @brief A FieldCollection is not move-assign-able, but move-construct-able
     */
    FieldCollection& operator=(FieldCollection&&) = delete;

    /**
     * @brief Checks if the collection contains a field with the given ID.
     *
     * @param id The ID of the field to check for.
     * @return true if the collection contains the field, false otherwise.
     */
    bool contains(const std::string& id) const;

    /**
     * @brief Inserts a field document into the collection.
     *
     * @param fd The field document to insert.
     * @return A string representing the unique identifier of the inserted field.
     */
    std::string insert(const FieldDocument& fd);

    /**
     * @brief Retrieves a field document by its ID.
     *
     * @param id The ID of the field document to retrieve.
     * @return FieldDocument& A reference to the field document.
     */
    FieldDocument& fieldDoc(const std::string& id);

    /**
     * @brief Retrieves a field document by its ID (const version).
     *
     * @param id The ID of the field document to retrieve.
     * @return const FieldDocument& A const reference to the field document.
     */
    const FieldDocument& fieldDoc(const std::string& id) const;

    /**
     * @brief Retrieves the instance of the FieldCollection with the given name.
     *
     * creates the FieldCollection if it does not exist.
     *
     * @param db The database to retrieve the FieldCollection from.
     * @param name The name of the FieldCollection.
     * @return FieldCollection& A reference to the FieldCollection.
     */
    static FieldCollection& instance(NeoFOAM::Database& db, std::string name);


    /**
     * @brief Retrieves the instance of the FieldCollection with the given name (const version).
     *
     * creates the FieldCollection if it does not exist.
     *
     * @param db The database to retrieve the FieldCollection from.
     * @param name The name of the FieldCollection.
     * @return const FieldCollection& A const reference to the FieldCollection.
     */
    static const FieldCollection& instance(const NeoFOAM::Database& db, std::string name);

    /**
     * @brief Retrieves the instance of the FieldCollection from a const registered FieldType
     *
     * @param field A registered FieldType
     * @return FieldCollection& A reference to the FieldCollection.
     */
    template<class FieldType>
    static FieldCollection& instance(FieldType& field)
    {
        validateRegistration(
            field, "attempting to retrieve FieldCollection from unregistered field"
        );
        return instance(field.db(), field.fieldCollectionName);
    }

    /**
     * @brief Retrieves the instance of the FieldCollection from a const registered FieldType
     *
     * @param field A registered FieldType
     * @return FieldCollection& A reference to the FieldCollection.
     */
    template<class FieldType>
    static const FieldCollection& instance(const FieldType& field)
    {
        validateRegistration(
            field, "attempting to retrieve FieldCollection from unregistered field"
        );
        const Database& db = field.db();
        const Collection& collection = db.at(field.fieldCollectionName);
        return collection.as<FieldCollection>();
        // return instance(field.db(), field.fieldCollectionName);
    }

    /**
     * @brief Registers a field in the collection.
     *
     * @tparam FieldType The type of the field to register.
     * @param createFunc The function to create the field document.
     * @return A reference to the registered field.
     */
    template<class FieldType>
    FieldType& registerField(CreateFunction createFunc)
    {
        FieldDocument doc = createFunc(db());
        if (!validateFieldDoc(doc.doc()))
        {
            throw std::runtime_error("Document is not valid");
        }

        std::string key = insert(doc);
        FieldDocument& fd = fieldDoc(key);
        FieldType& field = fd.field<FieldType>();
        field.key = key;
        field.fieldCollectionName = name();
        return field;
    }
};


/**
 * @brief Creates a FieldDocument from an existing field.
 *
 * This functor creates a FieldDocument from an existing field.
 *
 * @tparam FieldType The type of the field.
 * @param name The name of the field document.
 * @param field The field to create the document from.
 * @param timeIndex The time index of the field document.
 * @param iterationIndex The iteration index of the field document.
 * @param subCycleIndex The sub-cycle index of the field document.
 * @return The created FieldDocument.
 */
template<typename FieldType>
class CreateFromExistingField
{
public:

    std::string name;
    const FieldType& field;
    std::int64_t timeIndex = std::numeric_limits<std::int64_t>::max();
    std::int64_t iterationIndex = std::numeric_limits<std::int64_t>::max();
    std::int64_t subCycleIndex = std::numeric_limits<std::int64_t>::max();

    FieldDocument operator()(Database& db)
    {
        DomainField<typename FieldType::FieldValueType> domainField(
            field.mesh().exec(), field.internalField(), field.boundaryField()
        );

        FieldType vf(
            field.exec(), name, field.mesh(), domainField, field.boundaryConditions(), db, "", ""
        );

        if (field.registered())
        {
            const FieldCollection& fieldCollection = FieldCollection::instance(field);
            const FieldDocument& fieldDoc = fieldCollection.fieldDoc(field.key);
            if (timeIndex == std::numeric_limits<std::int64_t>::max())
            {
                timeIndex = fieldDoc.timeIndex();
            }
            if (iterationIndex == std::numeric_limits<std::int64_t>::max())
            {
                iterationIndex = fieldDoc.iterationIndex();
            }
            if (subCycleIndex == std::numeric_limits<std::int64_t>::max())
            {
                subCycleIndex = fieldDoc.subCycleIndex();
            }
        }
        return NeoFOAM::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            validateFieldDoc
        );
    }
};


} // namespace NeoFOAM
