// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "executorGenerator.hpp"


/** @brief a shortcut for  std::initializer_list */
template<typename T>
using I = std::initializer_list<T>;

/**
 * @brief Conditionally enters a Catch2 SECTION if the given condition is true.
 * @param COND The boolean condition to evaluate.
 * @param ... The section name and optional description forwarded to SECTION.
 */
#define SECTION_IF(COND, ...)                                                                      \
    if (COND) SECTION(__VA_ARGS__)

/**
 *  @brief Predicate for approximate floating-point comparison within a given margin.
 *
 * Used with @ref EqualsRangeMatcher and @ref IsEqualTo to compare scalar values
 * using Catch2's Approx mechanism, allowing for a configurable absolute tolerance.
 */
struct ApproxScalar
{
    NeoN::scalar margin; ///< Absolute tolerance margin for the comparison.

    /**
     * @brief Returns true if @p rhs and @p lhs are equal within the stored margin.
     * @param rhs Left-hand value passed to Catch::Approx.
     * @param lhs Right-hand value to compare against.
     */
    bool operator()(double rhs, double lhs) const
    {
        return Catch::Approx(rhs).margin(margin) == lhs;
    }
};

/** @brief Predicate for exact equality comparison using the built-in == operator.
 *
 * Used with @ref EqualsRangeMatcher and @ref IsEqualTo to compare integer or
 * other exactly-comparable types (e.g. NeoN::Vec3, NeoN::label).
 */
struct EqualInt
{
    /**
     * @brief Returns true if @p rhs and @p lhs are equal within the stored margin.
     * @param rhs Left-hand value passed to Catch::Approx.
     * @param lhs Right-hand value to compare against.
     */
    bool operator()(auto rhs, auto lhs) const { return rhs == lhs; }
};


/**
 * @brief Catch2 matcher that compares a NeoN device field against an expected std::vector.
 *
 * Copies the field to host on construction, then element-wise compares it against
 * the expected vector using the supplied predicate.
 *
 * @tparam Range     A NeoN field type that exposes @c copyToHost() and @c view().
 * @tparam Predicate A binary callable returning @c bool, e.g. @ref ApproxScalar or @ref EqualInt.
 */
template<typename Range, typename Predicate>
struct EqualsRangeMatcher : Catch::Matchers::MatcherGenericBase
{
    /**
     * @brief Constructs the matcher, copying @p range to host memory.
     * @param range The device field to compare against.
     * @param pred  The predicate used for element-wise comparison.
     */
    EqualsRangeMatcher(const Range range, Predicate pred) : range {range.copyToHost()}, pred_(pred)
    {}

    /** @brief Performs the element-wise comparison against @p other.
     * @tparam ValueType Element type of the expected vector.
     * @param other The expected values as a @c std::vector.
     * @return @c true if all elements compare equal under the stored predicate.
     */
    template<typename ValueType>
    bool match(const ValueType other) const
    {
        using std::begin;
        using std::end;
        return std::equal(begin(range.view()), end(range.view()), begin(other), end(other), pred_);
    }

    /** @brief Returns a human-readable description of the matcher for Catch2 failure
     * messages. */
    std::string describe() const override { return "!= " + Catch::rangeToString(range.view()); }

private:

    const Range range;     ///< Host copy of the device field.
    const Predicate pred_; ///< Predicate used for element-wise comparison.
};

/**
 * @brief Factory function that creates an @ref EqualsRangeMatcher for a NeoN field.
 *
 * Typical usage:
 * @code
 * REQUIRE_THAT(expected_vec, IsEqualTo(mesh.faceOwner(), EqualInt()));
 * REQUIRE_THAT(expected_vec, IsEqualTo(field, ApproxScalar(1e-15)));
 * @endcode
 *
 * @tparam Range     A NeoN field type that exposes @c copyToHost() and @c view().
 * @tparam Predicate A binary callable returning @c bool (default: @ref ApproxScalar).
 * @param range The device field to compare against.
 * @param pred  The predicate used for element-wise comparison (default:
 * ApproxScalar(1e-32)).
 * @return An @ref EqualsRangeMatcher configured with the given field and predicate.
 */
template<typename Range, typename Predicate = ApproxScalar>
auto IsEqualTo(const Range& range, Predicate pred = ApproxScalar(1e-32))
    -> EqualsRangeMatcher<Range, Predicate>
{
    return EqualsRangeMatcher<Range, Predicate> {range, pred};
}
