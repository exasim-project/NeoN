// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors
#pragma once

#include <limits>

#include <span>

namespace NeoN
{

/* @class Span
 *
 * @brief A wrapper class for std::span which allows to check whether the index access is in range
 * The Span can be initialized like a regular std::span or from an existing std::span
 *
 * @ingroup core
 *
 */
template<typename ValueType>
class View : public std::span<ValueType>
{
public:

    /* A flag to control whether the program should terminate on invalid memory access or throw.
     * Kokkos prefers to terminate, but for testing purpose the failureIndex is preferred
     */
    bool abort = true;

    /* a member to store the first out of range data access. This assumes a span has
     * at least a size of 1. A value of zero signals success. This is required we cannot
     * throw from a device function.
     */
    mutable size_t failureIndex = 0;

    using std::span<ValueType>::span; // Inherit constructors from std::span

    /* Constructor from existing std::span
     */
    View(std::span<ValueType> in) : View(in.begin(), in.end()) {}

    constexpr ValueType& operator[](std::size_t index) const
    {
#ifdef NF_DEBUG
        if (index >= this->size())
        {
            // TODO: currently this is failing on our AWS workflow, once we have clang>16 there
            // this should work again.
            // const std::string msg {"Index is out of range. Index: "} + to_string(index);
            if (abort)
            {
                Kokkos::abort("Index is out of range");
            }
            else
            {
                // NOTE: throwing from a device function does not work
                // throw std::invalid_argument("Index is out of range");
                if (failureIndex == 0)
                {
                    failureIndex = index;
                }
                return std::span<ValueType>::operator[](index);
            }
        }
#endif
        return std::span<ValueType>::operator[](index);
    }
};


} // namespace NeoN
