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
 * Used with @ref EqualsMatcher and @ref Equals to compare scalar values
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

/**
 * @brief Predicate for approximate comparison of NeoN::Vec3 values.
 *
 * Performs component-wise floating-point comparison using Catch2's
 * @ref Catch::Approx with a configurable absolute tolerance.
 *
 * This predicate is intended for use with @ref EqualsMatcher and
 * @ref Equals when comparing vector-valued fields (e.g. cell centres,
 * face centres, geometric quantities).
 *
 * Each component of the vector is compared independently using the same
 * absolute tolerance.
 *
 * Example usage:
 * @code
 * std::vector<NeoN::Vec3> expected = {
 *     {0.125, 0.5, 0.5},
 *     {0.375, 0.5, 0.5}
 * };
 *
 * REQUIRE_THAT(mesh.cellCentres(),
 *              Equals(expected, ApproxVec3{1e-12}));
 * @endcode
 */
struct ApproxVec3
{
    NeoN::scalar margin; ///< Absolute tolerance used for each component.

    /**
     * @brief Returns true if vectors @p rhs and @p lhs are approximately equal.
     *
     * The comparison is performed component-wise using @ref Catch::Approx:
     * \f[
     * |rhs_i - lhs_i| \le \text{margin}, \quad i = 0,1,2
     * \f]
     *
     * @param rhs Reference vector (used as the reference value in Catch::Approx).
     * @param lhs Vector to compare against.
     * @return @c true if all three components are approximately equal within
     *         the specified tolerance.
     */
    bool operator()(const NeoN::Vec3& rhs, const NeoN::Vec3& lhs) const
    {
        return Catch::Approx(rhs[0]).margin(margin) == lhs[0]
            && Catch::Approx(rhs[1]).margin(margin) == lhs[1]
            && Catch::Approx(rhs[2]).margin(margin) == lhs[2];
    }
};

/** @brief Predicate for exact equality comparison using the built-in == operator.
 *
 * Used with @ref EqualsMatcher and @ref Equals to compare integer or
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
 * @brief Catch2 matcher for element-wise comparison between an actual field and expected values.
 *
 * This matcher compares a NeoN field (or any compatible range) against an
 * expected container using a user-provided predicate. The comparison is
 * performed element-wise via @c std::equal.
 *
 * If the @p actual argument represents a device-resident field, it is first
 * copied to host memory via @c copyToHost() before comparison.
 *
 * Typical usage:
 * @code
 * std::vector<NeoN::Vec3> expected = { ... };
 *
 * REQUIRE_THAT(mesh.cellCentres(),
 *              Equals(expected, ApproxVec3{1e-12}));
 * @endcode
 *
 * @tparam Expected   Container type holding expected values (e.g. std::vector).
 * @tparam Predicate  Binary callable returning @c bool for comparing two elements
 *                    (e.g. @ref ApproxScalar, @ref ApproxVec3, @ref EqualInt).
 */
template<typename Expected, typename Predicate>
struct EqualsMatcher : Catch::Matchers::MatcherGenericBase
{
    /**
     * @brief Constructs the matcher with expected values and comparison predicate.
     *
     * @param expected Container holding the expected values.
     * @param pred     Predicate used for element-wise comparison.
     */
    EqualsMatcher(Expected expected, Predicate pred) : expected_(std::move(expected)), pred_(pred)
    {}

    /**
     * @brief Performs element-wise comparison against the @p actual argument.
     *
     * The @p actual object is assumed to be a NeoN field (or compatible type)
     * exposing @c copyToHost() and @c view(). The data is copied to host memory
     * and compared against @ref expected_ using @ref pred_.
     *
     * @tparam Actual Type of the actual object passed by Catch2.
     * @param actual  The object under test (typically a NeoN field).
     * @return @c true if all elements compare equal under the predicate,
     *         @c false otherwise.
     */
    template<typename Actual>
    bool match(const Actual& actual) const
    {
        using std::begin;
        using std::end;

        // Copy device field to host if needed
        auto actualHost = actual.copyToHost();

        return std::equal(
            begin(actualHost.view()),
            end(actualHost.view()),
            begin(expected_),
            end(expected_),
            pred_
        );
    }

    /**
     * @brief Returns a human-readable description of the matcher.
     *
     * This string is used by Catch2 when reporting assertion failures.
     *
     * @return Description of the expected values.
     */
    std::string describe() const override { return "equals " + Catch::rangeToString(expected_); }

private:

    Expected expected_; ///< Container holding expected values.
    Predicate pred_;    ///< Predicate used for element-wise comparison.
};


/**
 * @brief Factory function to create an @ref EqualsMatcher for element-wise comparison.
 *
 * This function constructs an @ref EqualsMatcher that compares an actual range
 * (e.g. NeoN field or container) against an expected container using a
 * user-defined predicate.
 *
 * It is intended for use with Catch2's @ref REQUIRE_THAT macro:
 * @code
 * REQUIRE_THAT(mesh.cellCentres(),
 *              Equals(expectedCentres, ApproxVec3{1e-12}));
 * @endcode
 *
 * If no predicate is provided, @ref ApproxScalar is used by default for
 * floating-point comparisons with a small tolerance.
 *
 * @tparam Expected   Type of the expected container (e.g. std::vector<T>).
 * @tparam Predicate  Binary predicate used for element-wise comparison.
 *                    Defaults to @ref ApproxScalar.
 *
 * @param expected    Container holding the expected values.
 * @param pred        Predicate used for comparison (default: ApproxScalar{1e-32}).
 *
 * @return An @ref EqualsMatcher configured with the provided expected values
 *         and comparison predicate.
 */
template<typename Expected, typename Predicate = ApproxScalar>
auto Equals(Expected expected, Predicate pred = Predicate {1e-32})
{
    return EqualsMatcher<Expected, Predicate> {std::move(expected), pred};
}

namespace Catch
{

/**
 * @brief Catch2 string conversion for NeoN::Vector.
 *
 * This specialization of @c Catch::StringMaker enables Catch2 to print
 * @c NeoN::Vector objects in assertion messages. Without this, Catch2
 * falls back to a placeholder ("{?}") because it does not know how to
 * convert the type to a string.
 *
 * The vector is first copied from device to host memory (if applicable)
 * using @c copyToHost(), and then converted to a string using
 * @c Catch::rangeToString.
 *
 * This allows assertions such as:
 * @code
 * REQUIRE_THAT(checkSparse, Equals(I({1.0, 5.0, 7.0, 8.0})));
 * @endcode
 * to produce informative output like:
 * @code
 * { 1.0, 5.0, 6.999999, 8.0 } is equal to { 1.0, 5.0, 7.0, 8.0 }
 * @endcode
 *
 * @tparam T Value type stored in the NeoN::Vector.
 */
template<typename T>
struct StringMaker<NeoN::Vector<T>>
{
    /**
     * @brief Converts a NeoN::Vector into a human-readable string.
     *
     * @param v The vector to convert (possibly device-resident).
     * @return A string representation of the vector contents.
     */
    static std::string convert(const NeoN::Vector<T>& v)
    {
        auto host = v.copyToHost();
        return rangeToString(host.view());
    }
};

} // namespace Catch
