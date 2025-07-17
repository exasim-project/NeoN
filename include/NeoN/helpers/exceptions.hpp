// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

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
}
