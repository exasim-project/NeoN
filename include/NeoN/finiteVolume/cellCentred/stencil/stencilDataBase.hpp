// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>
#include <any>
#include <string>

namespace NeoN
{

/**
 * @class StencilDataBase
 * @brief A class that represents a stencil database.
 *
 * The StencilDataBase class provides a container for storing stencil data. It
 * allows insertion, retrieval, and checking of stencil data using string keys.
 */
class StencilDataBase
{
public:

    /**
     * @brief Default constructor for StencilDataBase.
     */
    StencilDataBase() = default;

    /**
     * @brief Inserts a value into the stencil database.
     *
     * @tparam T The type of the value to be inserted.
     * @param key The key associated with the value.
     * @param value The value to be inserted.
     */
    template<typename T>
    void insert(const std::string& key, T value)
    {
        stencilDB_.emplace(key, value);
    }

    /**
     * @brief Retrieves a reference to the value associated with the given key.
     *
     * @param key The key associated with the value.
     * @return A reference to the value associated with the key.
     */
    std::any& operator[](const std::string& key);

    /**
     * @brief Retrieves a const reference to the value associated with the given
     * key.
     *
     * @param key The key associated with the value.
     * @return A const reference to the value associated with the key.
     */
    const std::any& operator[](const std::string& key) const;

    /**
     * @brief Retrieves a reference to the value associated with the given key.
     *
     * @tparam T The type of the value to be retrieved.
     * @param key The key associated with the value.
     * @return A reference to the value associated with the key.
     * @throws std::bad_any_cast if the value cannot be cast to type T.
     */
    template<typename T>
    T& get(const std::string& key)
    {
        return std::any_cast<T&>(stencilDB_.at(key));
    }

    /**
     * @brief Retrieves a const reference to the value associated with the given
     * key.
     *
     * @tparam T The type of the value to be retrieved.
     * @param key The key associated with the value.
     * @return A const reference to the value associated with the key.
     * @throws std::bad_any_cast if the value cannot be cast to type T.
     */
    template<typename T>
    const T& get(const std::string& key) const
    {
        return std::any_cast<const T&>(stencilDB_.at(key));
    }

    /**
     * @brief Checks if the stencil database contains a value associated with
     * the given key.
     *
     * @param key The key to check.
     * @return true if the database contains the key, false otherwise.
     */
    bool contains(const std::string& key) const;

private:

    /**
     * @brief The stencil database to register stencil data.
     */
    std::unordered_map<std::string, std::any> stencilDB_;
};

} // namespace NeoN
