// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <source_location>
#include <sstream>
#include <string_view>
#include "NeoN/core/primitives/label.hpp"

namespace NeoN
{
/**
 * @class Error
 * @brief Base class for consistent error representation.
 *
 * The Error class is used to report exceptional behaviour in library
 * functions. NeoN uses C++ exception mechanism to this end, and the
 * Error class represents a base class for all types of errors. The exact list
 * of errors which could occur during the execution of a certain library
 * routine is provided in the documentation of that routine, along with a short
 * description of the situation when that error can occur.
 * During runtime, these errors can be detected by using standard C++ try-catch
 * blocks, and a human-readable error description can be obtained by calling
 * the Error::what() method.
 *
 * @ingroup Error
 */
class Error : public std::exception
{
public:

    /**
     * Initializes an error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param what  The error message
     */
    Error(const std::string& file, int line, const std::string& what)
        : what_(file + ":" + std::to_string(line) + ": " + what)
    {}

    /**
     * Returns a human-readable string with a more detailed description of the
     * error.
     */
    virtual const char* what() const noexcept override { return what_.c_str(); }

private:

    const std::string what_;
};


/**
 * @class DimensionMismatch
 * @brief Error for handling two containers of incompatible lengths.
 *
 * DimensionMismatch is thrown if an operation is being applied to containers of
 * incompatible size.
 * @ingroup Error
 */
class DimensionMismatch : public Error
{
public:

    /**
     * Initializes a dimension mismatch error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The function name where the error occurred
     * @param length_a  The size of the first container
     * @param length_b  The size of the second container
     * @param clarification  An additional message describing the error further
     */
    DimensionMismatch(
        const std::string& file,
        int line,
        const std::string& func,
        localIdx lengthA,
        localIdx lengthB,
        const std::string& clarification
    )
        : Error(
            file,
            line,
            func + ": Trying to perform binary operation " + " " + std::to_string(lengthA) + ", "
                + std::to_string(lengthB) + " " + clarification
        )
    {}
};


/**
 * Asserts that `_op1` and `_op2` have the same length.
 *
 * @throw DimensionMismatch  if `_op1` and `_op2` differ in the number of
 *                           rows or columns
 */
#define NeoN_ASSERT_EQUAL_LENGTH(_op1, _op2)                                                       \
    auto s1 = static_cast<localIdx>(_op1.size());                                                  \
    auto s2 = static_cast<localIdx>(_op2.size());                                                  \
    if (s1 != s2)                                                                                  \
    {                                                                                              \
        throw ::NeoN::DimensionMismatch(                                                           \
            __FILE__, __LINE__, __func__, s1, s2, "expected equal dimensions"                      \
        );                                                                                         \
    }

/**
 * Checks that the half-open range [begin, end) lies inside a container of the given size.
 *
 * A range is valid if 0 <= begin <= end <= size. An empty range
 * (begin == end) is valid, including at begin == end == size.
 *
 * @param range  The pair {begin, end}, where end is exclusive
 * @param size   The size of the container being indexed
 *
 * @throw std::out_of_range  if begin < 0, begin > end, or end > size
 */
inline void validateRange(std::pair<localIdx, localIdx> range, size_t size)
{
    const auto begin = range.first;
    const auto end = range.second;

    if (begin < 0 || begin > end || static_cast<size_t>(end) > size)
    {
        throw std::out_of_range("The chosen range is not valid!");
    }
}

/**
 * Converts a size_t to localIdx, checking that the value fits.
 *
 * Use this instead of a plain static_cast when converting container sizes or
 * indices. A plain cast can silently wrap around if the value is larger than
 * the maximum localIdx.
 *
 * @param value  The unsigned value to convert, typically a size() result
 * @return The same value as a localIdx
 *
 * @throw std::length_error  if value is greater than the maximum localIdx
 */
inline localIdx toLocalIdx(size_t value)
{
    if (!std::in_range<localIdx>(value))
    {
        throw std::length_error("Value cannot be represented by localIdx!");
    }
    return static_cast<localIdx>(value);
}

inline void requireInput(
    bool condition,
    std::string_view message,
    const std::source_location location = std::source_location::current()
)
{
    if (condition)
    {
        return;
    }
    std::ostringstream error;
    error << "Invalid input: " << message << "\nFile: \n"
          << location.file_name() << "\nLine: " << location.line();

    throw NeoN::NeoNException(error.str());
}

} // namespace NeoN
