// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <string>
#include <functional>

#include "NeoN/core/demangle.hpp"
#include "NeoN/core/error.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/core/database/database.hpp"
#include "NeoN/core/database/collection.hpp"
#include "NeoN/core/database/document.hpp"

namespace NeoN::finiteVolume::cellCentred
{
/**
 * @brief Validates a VectorDocument.
 *
 * This function validates a VectorDocument by checking if it contains the required fields.
 *
 * @param doc The Document to validate.
 * @return true if the Document is valid, false otherwise.
 */
bool validateVectorDoc(const Document& doc);

/**
 * @class VectorDocument
 * @brief A class representing a field document in a database.
 *
 * The VectorDocument class represents a field document in a database. It is a subclass of the
 * Document class and provides additional functionality for accessing field-specific data.
 */
class VectorDocument
{
public:

    /**
     * @brief Constructs a VectorDocument with the given field and metadata.
     *
     * @tparam VectorType The type of the field.
     * @param field The field to store in the document.
     * @param timeIndex The time index of the field.
     * @param iterationIndex The iteration index of the field.
     * @param subCycleIndex The sub-cycle index of the field.
     */
    template<class VectorType>
    VectorDocument(
        const VectorType& field,
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
            validateVectorDoc
        )
    {}

    /**
     * @brief Constructs a VectorDocument with the given Document.
     *
     * @param doc The Document to construct the VectorDocument from.
     */
    VectorDocument(const Document& doc);

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
     * @tparam VectorType The type of the field.
     * @return A reference to the field.
     */
    template<class VectorType>
    VectorType& field()
    {
        return doc_.get<VectorType&>("field");
    }

    /**
     * @brief Retrieves the field from the document (const version).
     *
     * @tparam VectorType The type of the field.
     * @return A const reference to the field.
     */
    template<class VectorType>
    const VectorType& field() const
    {
        return doc_.get<const VectorType&>("field");
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
 * @brief A function type for creating a VectorDocument.
 *
 * This function type is used to create a VectorDocument and creates a
 * registered VectorType
 *
 * @param db The database to create the VectorDocument in.
 * @return The created VectorDocument.
 */
using CreateFunction = std::function<VectorDocument(NeoN::Database& db)>;

/**
 * @class VectorCollection
 * @brief A class representing a collection of field documents in a database.
 *
 * The VectorCollection class represents a collection of field documents in a database and provides
 * additional functionality for accessing field-specific data.
 */
class VectorCollection : public CollectionMixin<VectorDocument>
{
public:

    /**
     * @brief Constructs a VectorCollection with the given database and name.
     *
     * @param db The database to create the collection in.
     * @param name The name of the collection.
     */
    VectorCollection(NeoN::Database& db, std::string name);

    /**
     * @brief A VectorCollection is not copyable, but moveable
     */
    VectorCollection(const VectorCollection&) = delete;

    /**
     * @brief A VectorCollection is not copyable, but moveable
     */
    VectorCollection& operator=(const VectorCollection&) = delete;

    /**
     * @brief A VectorCollection is move constructable, but not copyable
     */
    VectorCollection(VectorCollection&&) = default;

    /**
     * @brief A VectorCollection is not move-assign-able, but move-construct-able
     */
    VectorCollection& operator=(VectorCollection&&) = delete;

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
    std::string insert(const VectorDocument& fd);

    /**
     * @brief Retrieves a field document by its ID.
     *
     * @param id The ID of the field document to retrieve.
     * @return VectorDocument& A reference to the field document.
     */
    VectorDocument& fieldDoc(const std::string& id);

    /**
     * @brief Retrieves a field document by its ID (const version).
     *
     * @param id The ID of the field document to retrieve.
     * @return const VectorDocument& A const reference to the field document.
     */
    const VectorDocument& fieldDoc(const std::string& id) const;

    /**
     * @brief Retrieves the instance of the VectorCollection with the given name.
     *
     * creates the VectorCollection if it does not exist.
     *
     * @param db The database to retrieve the VectorCollection from.
     * @param name The name of the VectorCollection.
     * @return VectorCollection& A reference to the VectorCollection.
     */
    static VectorCollection& instance(NeoN::Database& db, std::string name);


    /**
     * @brief Retrieves the instance of the VectorCollection with the given name (const version).
     *
     * creates the VectorCollection if it does not exist.
     *
     * @param db The database to retrieve the VectorCollection from.
     * @param name The name of the VectorCollection.
     * @return const VectorCollection& A const reference to the VectorCollection.
     */
    static const VectorCollection& instance(const NeoN::Database& db, std::string name);

    /**
     * @brief Retrieves the instance of the VectorCollection from a const registered VectorType
     *
     * @param field A registered VectorType
     * @return VectorCollection& A reference to the VectorCollection.
     */
    template<class VectorType>
    static VectorCollection& instance(VectorType& field)
    {
        validateRegistration(
            field, "attempting to retrieve VectorCollection from unregistered field"
        );
        return instance(field.db(), field.fieldCollectionName);
    }

    /**
     * @brief Retrieves the instance of the VectorCollection from a const registered VectorType
     *
     * @param field A registered VectorType
     * @return VectorCollection& A reference to the VectorCollection.
     */
    template<class VectorType>
    static const VectorCollection& instance(const VectorType& field)
    {
        validateRegistration(
            field, "attempting to retrieve VectorCollection from unregistered field"
        );
        const Database& db = field.db();
        const Collection& collection = db.at(field.fieldCollectionName);
        return collection.as<VectorCollection>();
        // return instance(field.db(), field.fieldCollectionName);
    }

    /**
     * @brief Registers a field in the collection.
     *
     * @tparam VectorType The type of the field to register.
     * @param createFunc The function to create the field document.
     * @return A reference to the registered field.
     */
    template<class VectorType>
    VectorType& registerVector(CreateFunction createFunc)
    {
        VectorDocument doc = createFunc(db());
        if (!validateVectorDoc(doc.doc()))
        {
            throw std::runtime_error {"Document is not valid"};
        }

        std::string key = insert(doc);
        VectorDocument& fd = fieldDoc(key);
        VectorType& field = fd.field<VectorType>();
        field.key = key;
        field.fieldCollectionName = name();
        return field;
    }
};


/**
 * @brief Creates a VectorDocument from an existing field.
 *
 * This functor creates a VectorDocument from an existing field.
 *
 * @tparam VectorType The type of the field.
 * @param name The name of the field document.
 * @param field The field to create the document from.
 * @param timeIndex The time index of the field document.
 * @param iterationIndex The iteration index of the field document.
 * @param subCycleIndex The sub-cycle index of the field document.
 * @return The created VectorDocument.
 */
template<typename VectorType>
class CreateFromExistingVector
{
public:

    std::string name;
    const VectorType& field;
    std::int64_t timeIndex = std::numeric_limits<std::int64_t>::max();
    std::int64_t iterationIndex = std::numeric_limits<std::int64_t>::max();
    std::int64_t subCycleIndex = std::numeric_limits<std::int64_t>::max();

    VectorDocument operator()(Database& db)
    {
        Field<typename VectorType::VectorValueType> domainVector(
            field.mesh().exec(), field.internalVector(), field.boundaryData()
        );

        VectorType vf(
            field.exec(), name, field.mesh(), domainVector, field.boundaryConditions(), db, "", ""
        );

        if (field.registered())
        {
            const VectorCollection& fieldCollection = VectorCollection::instance(field);
            const VectorDocument& fieldDoc = fieldCollection.fieldDoc(field.key);
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
        return NeoN::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            validateVectorDoc
        );
    }
};


} // namespace NeoN
