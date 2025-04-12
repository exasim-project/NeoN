// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <unordered_map>
#include <string>
#include <memory>

#include "NeoFOAM/core/database/collection.hpp"

namespace NeoFOAM
{


class Database
{
public:

    /**
     * @brief Inserts a collection into the database with the specified key.
     *
     * @param key The key associated with the collection to be inserted.
     * @param col The collection to be inserted into the database.
     * @return Collection& A reference to the inserted collection.
     */
    Collection& insert(const std::string& key, Collection&& col);

    /**
     * @brief Checks if the database contains an collection with the specified name.
     *
     * @param name The name of the collection to check for.
     * @return true if the collection exists in the database, false otherwise.
     */
    bool contains(const std::string& name) const;

    /**
     * @brief Removes a collection from the database.
     *
     * This function removes the collection with the specified name from the database.
     *
     * @param name The name of the collection to be removed.
     * @return true if the collection was successfully removed, false otherwise.
     */
    bool remove(const std::string& name);

    /**
     * @brief Retrieves a collection by its name.
     *
     * This function searches for and returns a reference to a collection
     * identified by the given name. If the collection does not exist,
     * the behavior is undefined.
     *
     * @param name The name of the collection to retrieve.
     * @return Collection& A reference to the collection with the specified name.
     */
    Collection& at(const std::string& name);

    /**
     * @brief Retrieves a collection by its name (const version).
     *
     * This function searches for and returns a const reference to a collection
     * identified by the given name. If the collection does not exist,
     * the behavior is undefined.
     *
     * @param name The name of the collection to retrieve.
     * @return const Collection& A const reference to the collection with the specified name.
     */
    const Collection& at(const std::string& name) const;

    /**
     * @brief Retrieves a collection by its name and casts it to the specified type.
     *
     * This function retrieves a collection by its name and attempts to cast it to the specified
     * type. If the collection does not exist or the cast fails, an exception is thrown.
     *
     * @tparam CollectionType The type to cast the collection to.
     * @param name The name of the collection to retrieve.
     * @return CollectionType& A reference to the collection cast to the specified type.
     */
    template<typename CollectionType = Collection>
    CollectionType& at(const std::string& name)
    {
        Collection& collection = at(name);
        return collection.as<CollectionType>();
    }


    /**
     * @brief Returns the size of the database.
     *
     * This function provides the number of elements currently stored in the database.
     *
     * @return std::size_t The number of elements in the database.
     */
    std::size_t size() const;


private:

    /**
     * @brief A map that associates collection names with their corresponding Collection objects.
     *
     * This unordered map uses strings as keys to represent the names of the collections,
     * and the values are Collection objects
     */
    std::unordered_map<std::string, Collection> collections_;
};


/**
 * @brief Validates that a field is registered in the database.
 *
 * @tparam Type The type of the field.
 * @param field The field to validate.
 * @throws std::runtime_error if the field is not registered in the database.
 */
template<typename Type>
void validateRegistration(const Type& obj, const std::string errorMessage)
{
    if (!obj.registered())
    {
        throw std::runtime_error(errorMessage);
    }
}

} // namespace NeoFOAM
