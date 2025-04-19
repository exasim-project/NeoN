// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors
#pragma once

#include <limits>
#include <span>
#include <type_traits>

#include "NeoN/core/primitives/label.hpp"

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

    using base = std::span<ValueType>;

    /* A flag to control whether the program should terminate on invalid memory access or throw.
     * Kokkos prefers to terminate, but for testing purpose the failureIndex is preferred
     */
    bool abort = true;

    /* a member to store the first out of range data access. This assumes a span has
     * at least a size of 1. A value of zero signals success. This is required we cannot
     * throw from a device function.
     */
    mutable localIdx failureIndex = 0;

    using std::span<ValueType>::span; // Inherit constructors from std::span

    /* Constructor from existing std::span
     */
    View(std::span<ValueType> in) : View(in.begin(), in.end()) {}

    constexpr ValueType& operator[](localIdx index) const
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
                return std::span<ValueType>::operator[](static_cast<size_t>(index));
            }
        }
#endif
        return std::span<ValueType>::operator[](static_cast<size_t>(index));
    }

    localIdx size() const { return static_cast<localIdx>(base::size()); }

    View<ValueType> subspan(localIdx start, localIdx length) const
    {
        return base::subspan(static_cast<size_t>(start), static_cast<size_t>(length));
    }

    View<ValueType> subspan(localIdx start) const
    {
        return base::subspan(static_cast<size_t>(start));
    }
};

/**
 * @brief Concept, for any type which has the 'view' method.
 * @tparam Types Class type with potential 'view' method.
 */
template<class Type>
concept hasView =
    requires(Type& inst) { inst.view(); } || requires(const Type& inst) { inst.view(); };

/**
 * @brief Unpacks all views of the passed classes.
 * @tparam Types Types of the classes with views
 * @return Tuple containing the unpacked views (use structured bindings).
 */
template<typename... Types>
    requires(hasView<std::remove_reference_t<Types>> && ...)
auto views(Types&... args)
{
    return std::tuple(args.view()...);
}

} // namespace NeoN
