// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <any>

#include "NeoN/core/demangle.hpp"

namespace NeoN
{


void logOutRange(
    const std::out_of_range& e, const std::size_t& key, const std::vector<std::any>& data
);


/**
 * @class TokenList
 * @brief A class representing a list of tokens.
 *
 * The TokenList class provides functionality to store and manipulate a list of tokens.
 * It supports insertion, removal, and retrieval of tokens of any type using std::any.
 */
class TokenList
{
public:

    TokenList();

    TokenList(const TokenList&);

    /**
     * @brief Construct a TokenList object from a vector of std::any.
     * @param data A vector of std::any.
     */
    TokenList(const std::vector<std::any>& data, size_t nextIndex = 0);

    /**
     * @brief Construct a TokenList object from an initializer list of std::any.
     * @param initList An initializer list of std::any.
     */
    TokenList(const std::initializer_list<std::any>& initList);

    /**
     * @brief Inserts a value into the token list.
     * @param value The value to insert.
     */
    void insert(const std::any& value);

    /**
     * @brief Removes a value from the token list based on the specified index.
     * @param index The index of the value to remove.
     */
    void remove(size_t index);

    /**
     * @brief Checks if the token list is empty.
     * @return True if the token list is empty, false otherwise.
     */
    [[nodiscard]] bool empty() const;

    /**
     * @brief Removes first entry of TokenList and returns it.
     *
     * @tparam T The type to cast the value to.
     * @return The first value.
     */
    template<typename ReturnType>
    ReturnType popFront()
    {
        ReturnType ret {get<ReturnType>(0)};
        data_.erase(data_.begin());
        return ret;
    }

    /**
     * @brief Retrieves the size of the token list.
     * @return The size of the token list.
     */
    [[nodiscard]] size_t size() const;

    /**
     * @brief Retrieves the value associated with the given index, casting it to
     * the specified type.
     * @tparam T The type to cast the value to.
     * @param idx The index to retrieve the value for.
     * @return A reference to the value associated with the index, casted to
     * type T.
     */
    template<typename ReturnType>
    [[nodiscard]] ReturnType& get(const size_t& idx)
    {
        try
        {
            return std::any_cast<ReturnType&>(data_.at(idx));
        }
        catch (const std::bad_any_cast& e)
        {
            logBadAnyCast<ReturnType>(e, idx, data_);
            throw e;
        }
        catch (const std::out_of_range& e)
        {
            logOutRange(e, idx, data_);
            throw e;
        }
    }

    /**
     * @brief Retrieves the value associated with the nextIndex, casting it to
     * the specified type.
     * @tparam T The type to cast the value to.
     * @return A reference to the value associated with the index, casted to
     * type T.
     */
    template<typename ReturnType>
    ReturnType& next()
    {
        ReturnType& retValue = get<ReturnType&>(nextIndex_);
        nextIndex_++;
        return retValue;
    }

    /**
     * @brief Retrieves the value associated with the given index, casting it to
     * the specified type.
     * @tparam T The type to cast the value to.
     * @param idx The index to retrieve the value for.
     * @return A const reference to the value associated with the index, casted to
     * type T.
     */
    template<typename ReturnType>
    [[nodiscard]] const ReturnType& get(const size_t& idx) const
    {
        try
        {
            return std::any_cast<const ReturnType&>(data_.at(idx));
        }
        catch (const std::bad_any_cast& e)
        {
            logBadAnyCast<ReturnType>(e, idx, data_);
            throw e;
        }
        catch (const std::out_of_range& e)
        {
            logOutRange(e, idx, data_);
            throw e;
        }
    }

    /**
     * @brief Retrieves the value associated with the nextIndex, casting it to
     * the specified type.
     * @tparam T The type to cast the value to.
     * @return A const reference to the value associated with the index, casted to
     * type T.
     */
    template<typename ReturnType>
    const ReturnType& next() const
    {
        const ReturnType& retValue = get<ReturnType>(nextIndex_);
        nextIndex_++;
        return retValue;
    }

    /**
     * @brief Retrieves the value associated with the given index.
     * @param idx The index to retrieve the value for.
     * @return A reference to the value associated with the index.
     */
    [[nodiscard]] std::any& operator[](const size_t& idx);

    [[nodiscard]] std::vector<std::any>& tokens();


private:

    std::vector<std::any> data_;
    mutable size_t nextIndex_;
};

} // namespace NeoN
