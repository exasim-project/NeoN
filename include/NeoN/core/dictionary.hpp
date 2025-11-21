// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>
#include <any>
#include <string>
#include <vector>

#include "NeoN/core/demangle.hpp"

namespace NeoN
{

void logOutRange(
    const std::out_of_range& e,
    const std::string& key,
    const std::unordered_map<std::string, std::any>& data
);

/**
 * @class Dictionary
 * @brief A class representing a dictionary that stores key-value pairs.
 *
 * The Dictionary class provides a way to store and retrieve values using
 * string. It supports inserting key-value pairs, accessing values using the
 * subscript operator, and retrieving values of specific types using the `get`
 * function. It also supports storing sub-dictionaries, which can be accessed
 * using the `subDict` function. The values are stored using `std::any`, which
 * allows storing values of any type.
 */
class Dictionary
{
public:

    Dictionary() = default;

    Dictionary(const std::unordered_map<std::string, std::any>& keyValuePairs);

    Dictionary(const std::initializer_list<std::pair<std::string, std::any>>& initList);

    /**
     * @brief Inserts a key-value pair into the dictionary.
     * @param key The key to insert.
     * @param value The value to insert.
     */
    void insert(const std::string& key, const std::any& value);

    /**
     * @brief Checks if the given key is present in the dictionary.
     * @param key The key to check.
     * @return True if the key is present, false otherwise.
     */
    [[nodiscard]] bool contains(const std::string& key) const;

    /**
     * @brief Removes an entry from the dictionary based on the specified key.
     *
     * This function removes the entry with the specified key from the
     * dictionary.
     *
     * @param key The key of the entry to be removed.
     */
    void remove(const std::string& key);

    /**
     * @brief Accesses the value associated with the given key.
     * @param key The key to access.
     * @return A reference to the value associated with the key.
     */
    [[nodiscard]] std::any& operator[](const std::string& key);

    /**
     * @brief Accesses the value associated with the given key.
     * @param key The key to access.
     * @return A const reference to the value associated with the key.
     */
    [[nodiscard]] const std::any& operator[](const std::string& key) const;

    /**
     * @brief Retrieves the value associated with the given key, casting it to
     * the specified type.
     * @tparam T The type to cast the value to.
     * @param key The key to retrieve the value for.
     * @return A reference to the value associated with the key, casted to type
     * T.
     */
    template<typename T>
    [[nodiscard]] T& get(const std::string& key)
    {
        try
        {
            return std::any_cast<T&>(operator[](key));
        }
        catch (const std::bad_any_cast& e)
        {
            logBadAnyCast<T>(e, key, data_);
            throw;
        }
    }

    /**
     * @brief Retrieves the value associated with the given key, casting it to
     * the specified type.
     * @tparam T The type to cast the value to.
     * @param key The key to retrieve the value for.
     * @return A const reference to the value associated with the key, casted to
     * type T.
     */
    template<typename T>
    [[nodiscard]] const T& get(const std::string& key) const
    {
        try
        {
            return std::any_cast<const T&>(operator[](key));
        }
        catch (const std::bad_any_cast& e)
        {
            logBadAnyCast<T>(e, key, data_);
            throw;
        }
    }

    /**
     * @brief Checks if the value associated with the given key is of given Type T.
     * @param key The key to check.
     * @return True if the value is a dictionary, false otherwise.
     */
    template<typename T>
    [[nodiscard]] bool isType(const std::string& key) const
    {
        return contains(key) && std::any_cast<T>(&data_.at(key));
    }

    /**
     * @brief Checks if the value associated with the given key is a dictionary.
     * @param key The key to check.
     * @return True if the value is a dictionary, false otherwise.
     */
    [[nodiscard]] bool isDict(const std::string& key) const;


    /**
     * @brief Retrieves a sub-dictionary associated with the given key.
     * @param key The key to retrieve the sub-dictionary for.
     * @return A reference to the sub-dictionary associated with the key.
     */
    Dictionary& subDict(const std::string& key);

    /**
     * @brief Retrieves a sub-dictionary associated with the given key.
     * @param key The key to retrieve the sub-dictionary for.
     * @return A reference to the sub-dictionary associated with the key.
     */
    const Dictionary& subDict(const std::string& key) const;

    /**
     * @brief Retrieves the keys of the dictionary.
     * @return A vector containing the keys of the dictionary.
     */
    std::vector<std::string> keys() const;

    /**
     * @brief Retrieves the underlying unordered map of the dictionary.
     * @return A reference to the underlying unordered map.
     */
    std::unordered_map<std::string, std::any>& getMap();

    /**
     * @brief Retrieves the underlying unordered map of the dictionary.
     * @return A const reference to the underlying unordered map.
     */
    const std::unordered_map<std::string, std::any>& getMap() const;

    /**
     * @brief Checks whether the dictionary is empty
     * @return A bool indicating if the dictionary is empty
     */
    bool empty() const { return data_.empty(); }

private:

    std::unordered_map<std::string, std::any> data_;
};

std::ostream& operator<<(std::ostream& os, const Dictionary& in);

} // namespace NeoN
