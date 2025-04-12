// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include <algorithm> // for std::sort

#include "NeoN/core/database/document.hpp"

namespace NeoN
{


// forward declaration
class Database;

/**
 * @class Collection
 * @brief A type-erased interface collection types.
 *
 * The Collection class provides a type-erased interface for different collection types.
 * It allows operations such as retrieving documents, finding documents based on predicates,
 * and accessing collection metadata.
 */
class Collection
{
public:

    /**
     * @brief Constructs a Collection from a specific collection type.
     *
     * @tparam CollectionType The type of the collection to be wrapped.
     * @param collection The collection instance to be wrapped.
     */
    template<typename CollectionType>
    Collection(CollectionType collection)
        : impl_(std::make_unique<CollectionModel<CollectionType>>(std::move(collection)))
    {}

    /**
     * @brief A Collection is not copyable, only moveable.
     */
    Collection(const Collection& other) = delete;

    /**
     * @brief A Collection is not copyable, only moveable.
     */
    Collection& operator=(const Collection& other) = delete;

    /**
     * @brief A Collection is moveable, but not copyable
     */
    Collection(Collection&& other) = default;

    /**
     * @brief A Collection is moveable, but not copyable
     */
    Collection& operator=(Collection&& other) = default;

    /**
     * @brief Retrieves a document by its ID.
     *
     * @param id The ID of the document to retrieve.
     * @return Document& A reference to the document.
     */
    Document& doc(const std::string& id);

    /**
     * @brief Retrieves a document by its ID (const version).
     *
     * @param id The ID of the document to retrieve.
     * @return const Document& A const reference to the document.
     */
    const Document& doc(const std::string& id) const;

    /**
     * @brief Finds documents that match a given predicate.
     *
     * @param predicate A function that takes a Document and returns true if it matches the
     * criteria.
     * @return std::vector<std::string> A vector of document IDs that match the predicate.
     */
    std::vector<std::string> find(const std::function<bool(const Document&)>& predicate) const;

    /**
     * @brief Returns the number of documents in the collection.
     *
     * @return size_t The number of documents.
     */
    size_t size() const;

    /**
     * @brief Returns the type of the collection.
     *
     * @return std::string The type of the collection.
     */
    std::string type() const;

    /**
     * @brief Returns the name of the collection.
     *
     * @return std::string The name of the collection.
     */
    std::string name() const;

    /**
     * @brief Returns a reference to the database containing the collection.
     *
     * @return Database& A reference to the database.
     */
    Database& db();

    /**
     * @brief Returns a const reference to the database containing the collection.
     *
     * @return const Database& A const reference to the database.
     */
    const Database& db() const;

    /**
     * @brief Casts the collection to a specific collection type.
     *
     * @tparam CollectionType The type to cast to.
     * @return CollectionType& A reference to the casted collection.
     * @throws std::bad_cast if the cast fails.
     */
    template<typename CollectionType>
    CollectionType& as()
    {
        auto derived = dynamic_cast<CollectionModel<CollectionType>*>(impl_.get());
        if (!derived)
        {
            throw std::bad_cast();
        }
        return derived->collection_;
    }


    /**
     * @brief Casts the collection to a specific collection type (const version).
     *
     * @tparam CollectionType The type to cast to.
     * @return const CollectionType& A const reference to the casted collection.
     * @throws std::bad_cast if the cast fails.
     */
    template<typename CollectionType>
    const CollectionType& as() const
    {
        auto derived = dynamic_cast<CollectionModel<CollectionType>*>(impl_.get());
        if (!derived)
        {
            throw std::bad_cast();
        }
        return derived->collection_;
    }

private:

    struct CollectionConcept
    {
        virtual ~CollectionConcept() = default;
        virtual Document& doc(const std::string& id) = 0;
        virtual const Document& doc(const std::string& id) const = 0;
        virtual std::vector<std::string> find(const std::function<bool(const Document&)>& predicate
        ) const = 0;
        virtual size_t size() const = 0;
        virtual std::string type() const = 0;
        virtual std::string name() const = 0;
        virtual Database& db() = 0;
        virtual const Database& db() const = 0;
    };

    template<typename CollectionType>
    struct CollectionModel : CollectionConcept
    {
        CollectionModel(CollectionType collection) : collection_(std::move(collection)) {}

        Document& doc(const std::string& id) override { return collection_.doc(id); }

        const Document& doc(const std::string& id) const override { return collection_.doc(id); }

        std::vector<std::string> find(const std::function<bool(const Document&)>& predicate
        ) const override
        {
            return collection_.find(predicate);
        }

        size_t size() const override { return collection_.size(); }

        std::string type() const override { return collection_.type(); }

        std::string name() const override { return collection_.name(); }

        Database& db() override { return collection_.db(); }

        const Database& db() const override { return collection_.db(); }

        CollectionType collection_;
    };

    std::unique_ptr<CollectionConcept> impl_;
};

/**
 * @class CollectionMixin
 * @brief A mixin class for collection of documents in a database to simplify the implementation of
 * common operations.
 * @tparam DocumentType The type of documents stored in the collection.
 */
template<typename DocumentType>
class CollectionMixin
{

public:

    /**
     * @brief Constructs a CollectionMixin with the given database and collection name.
     *
     * @param db The database reference.
     * @param name The name of the collection.
     */
    CollectionMixin(NeoN::Database& db, std::string name) : docs_(), db_(db), name_(name) {}

    /**
     * @biref A CollectionMixin is not copyable, only moveable.
     */
    CollectionMixin(const CollectionMixin&) = delete;

    /**
     * @biref A CollectionMixin is not copyable, only moveable.
     */
    CollectionMixin& operator=(const CollectionMixin&) = delete;

    /**
     * @biref A CollectionMixin is moveable, but not copyable.
     */
    CollectionMixin(CollectionMixin&&) = default;

    /**
     * @biref A CollectionMixin is not move-assign-able.
     */
    CollectionMixin& operator=(CollectionMixin&&) = delete;

    /**
     * @brief Retrieves a document by its ID.
     *
     * @param id The ID of the document.
     * @return Document& A reference to the document.
     */
    Document& doc(const std::string& id) { return docs_.at(id).doc(); }

    /**
     * @brief Retrieves a document by its ID (const version).
     *
     * @param id The ID of the document.
     * @return const Document& A const reference to the document.
     */
    const Document& doc(const std::string& id) const { return docs_.at(id).doc(); }

    /**
     * @brief Finds documents that match a given predicate.
     *
     * @param predicate A function that takes a const reference to a Document and returns a bool.
     * @return std::vector<std::string> A vector of document IDs that match the predicate.
     */
    std::vector<std::string> find(const std::function<bool(const Document&)>& predicate) const
    {
        std::vector<std::string> result;
        for (const auto& [key, doc] : docs_)
        {
            if (predicate(doc.doc()))
            {
                result.push_back(doc.id());
            }
        }
        return result;
    }

    /**
     * @brief Gets the number of documents in the collection.
     *
     * @return std::size_t The number of documents.
     */
    std::size_t size() const { return docs_.size(); }

    /**
     * @brief Gets a const reference to the database.
     *
     * @return const NeoN::Database& A const reference to the database.
     */
    const NeoN::Database& db() const { return db_; }

    /**
     * @brief Gets a reference to the database.
     *
     * @return NeoN::Database& A reference to the database.
     */
    NeoN::Database& db() { return db_; }

    /**
     * @brief Gets the name of the collection.
     *
     * @return const std::string& A const reference to the collection name.
     */
    const std::string& name() const { return name_; }

    /**
     * @brief Gets the type name of the documents in the collection.
     *
     * @return std::string The type name of the documents.
     */
    std::string type() const { return DocumentType::typeName(); }

    /**
     * @brief Gets the sorted keys of the documents in the collection.
     *
     * @return std::vector<std::string> A vector of sorted document keys.
     */
    std::vector<std::string> sortedKeys() const
    {
        std::vector<std::string> result;
        for (const auto& [key, doc] : docs_)
        {
            result.push_back(key);
        }
        std::sort(result.begin(), result.end());
        return result;
    }

protected:

    std::unordered_map<std::string, DocumentType> docs_; ///< The map of document IDs to documents.
    NeoN::Database& db_;                                 ///< The reference to the database.
    std::string name_;                                   ///< The name of the collection.
};

} // namespace NeoN
