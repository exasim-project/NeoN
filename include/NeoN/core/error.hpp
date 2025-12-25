// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <sstream>
#include <iostream>

#ifdef NF_WITH_MPI_SUPPORT
#include <mpi.h>
#endif

// compiling with clang and cuda fails to
// find source location
// #include <source_location>
// #include <experimental/source_location>

// #include "logging.hpp"
#include "info.hpp"

#ifdef NF_DEBUG_MESSAGING
#include "cpptrace/cpptrace.hpp"
#endif


namespace NeoN
{

/**
 * @brief Custom exception class for NeoN.
 *
 * This class is derived from std::exception and provides a custom exception type for NeoN. It
 * stores an error message and overrides the what() function to return the error message.
 */
class NeoNException : public std::exception
{
public:

    /**
     * @brief Constructs a NeoNException object with the given error message.
     * @param message The error message associated with the exception.
     */
    explicit NeoNException(const std::string& message) : message_(message) {}

    /**
     * @brief Returns the error message associated with the exception.
     * @return const char* The error message.
     */
    const char* what() const noexcept override { return message_.c_str(); }

private:

    std::string message_; /**< The error message associated with the exception. */
};

} // namespace NeoN

#ifdef NF_DEBUG_MESSAGING
/**
 * @def NF_ERROR_MESSAGE
 * @brief Macro for generating an error message with debug information.
 *
 * This macro generates an error message with the specified message, current file name, current line
 * number, and debug trace information (if available).
 *
 * @param message The error message to be included in the generated message.
 * @return std::string The generated error message.
 */
#define NF_ERROR_MESSAGE(message)                                                                  \
    "Error: " << message << "\nFile: " << __FILE__ << "\nLine: " << __LINE__ << "\n"               \
              << cpptrace::generate_trace().to_string() << "\n"
#else
/**
 * @def NF_ERROR_MESSAGE
 * @brief Macro for generating an error message without debug information.
 *
 * This macro generates an error message with the specified message, current file name, and current
 * line number.
 *
 * @param message The error message to be included in the generated message.
 * @return std::string The generated error message.
 */
#define NF_ERROR_MESSAGE(message)                                                                  \
    "Error: " << message << "\nFile: " << __FILE__ << "\nLine: " << __LINE__ << "\n"
#endif

/**
 * @def NF_ERROR_EXIT
 * @brief Macro for printing an error message and aborting the program.
 *
 * This macro prints the specified error message to the standard error stream, including the current
 * file name and current line number, and then calls std::abort() to terminate the program.
 *
 * @param message The error message to be printed.
 */

#ifdef NF_WITH_MPI_SUPPORT
#define NF_ERROR_EXIT(message)                                                                     \
    do                                                                                             \
    {                                                                                              \
        std::cout << message << "\n";                                                              \
        cpptrace::generate_trace().print();                                                        \
        MPI_Abort(MPI_COMM_WORLD, 1);                                                              \
    }                                                                                              \
    while (false)
#else
#define NF_ERROR_EXIT(message)                                                                     \
    do                                                                                             \
    {                                                                                              \
        std::cerr << NF_ERROR_MESSAGE(message);                                                    \
        std::exit(1);                                                                              \
    }                                                                                              \
    while (false)
#endif

/**
 * @def NF_THROW
 * @brief Macro for throwing a NeoNException with the specified error message.
 *
 * This macro constructs a std::stringstream to concatenate the specified error message with the
 * current file name and current line number, and then throws a NeoNException with the resulting
 * string as the error message.
 *
 * @param message The error message to be included in the exception.
 */
#define NF_THROW(message)                                                                          \
    throw NeoN::NeoNException((std::stringstream() << NF_ERROR_MESSAGE(std::string(message))).str())

/**
 * @def NF_ASSERT
 * @brief Macro for asserting a condition and printing an error message if the
 * condition is false.
 *
 * This macro checks the specified condition and, if it evaluates to false, prints an error message
 * to the standard error stream, including the current file name and current line number, and then
 * calls std::abort() to terminate the program.
 *
 * @param condition The condition to be checked.
 * @param message The error message to be printed if the condition is false.
 */
#define NF_ASSERT(condition, message)                                                              \
    do                                                                                             \
    {                                                                                              \
        if (!(condition)) [[unlikely]]                                                             \
        {                                                                                          \
            NF_ERROR_EXIT("Assertion `" #condition "` failed.\n       " << message);               \
        }                                                                                          \
    }                                                                                              \
    while (false)

/**
 * @def NF_ASSERT_THROW
 * @brief Macro for asserting a condition and throwing a NeoNException if the condition is false.
 *
 * This macro checks the specified condition and, if it evaluates to false, constructs a
 * std::stringstream to concatenate the specified error message with the current file name and
 * current line number, and then throws a NeoNException with the resulting string as the error
 * message.
 *
 * @param condition The condition to be checked.
 * @param message The error message to be included in the exception if the
 * condition is false.
 */
#define NF_ASSERT_THROW(condition, message)                                                        \
    do                                                                                             \
    {                                                                                              \
        if (!(condition)) [[unlikely]]                                                             \
        {                                                                                          \
            NF_THROW("Assertion `" #condition "` failed.\n       " << message);                    \
        }                                                                                          \
    }                                                                                              \
    while (false)

#ifdef NF_DEBUG
/**
 * @def NF_DEBUG_ASSERT
 * @brief Macro for asserting a condition and printing an error message if the condition is false
 * (only in debug mode).
 *
 * This macro is equivalent to NF_ASSERT in debug mode.
 *
 * @param condition The condition to be checked.
 * @param message The error message to be printed if the condition is false.
 */
#define NF_DEBUG_ASSERT(condition, message) NF_ASSERT(condition, message)

/**
 * @def NF_DEBUG_ASSERT_THROW
 * @brief Macro for asserting a condition and throwing a NeoNException if the condition is false
 * (only in debug mode).
 *
 * This macro is equivalent to NF_ASSERT_THROW in debug mode.
 *
 * @param condition The condition to be checked.
 * @param message The error message to be included in the exception if the
 * condition is false.
 */
#define NF_DEBUG_ASSERT_THROW(condition, message) NF_ASSERT_THROW(condition, message)
#else
/**
 * @def NF_DEBUG_ASSERT
 * @brief Macro for asserting a condition and printing an error message if the condition is false
 * (only in debug mode).
 *
 * This macro does nothing in release mode.
 *
 * @param condition The condition to be checked.
 * @param message The error message to be printed if the condition is false.
 */
#define NF_DEBUG_ASSERT(condition, message) ((void)0)

/**
 * @def NF_DEBUG_ASSERT_THROW
 * @brief Macro for asserting a condition and throwing a NeoNException if the condition is false
 * (only in debug mode).
 *
 * This macro does nothing in release mode.
 *
 * @param condition The condition to be checked.
 * @param message The error message to be included in the exception if the
 * condition is false.
 */
#define NF_DEBUG_ASSERT_THROW(condition, message) ((void)0)
#endif

/**
 * @def NF_ASSERT_EQUAL
 * @brief Macro for asserting that two values are equal and printing an error message if they are
 * not.
 *
 * This macro checks that the two specified values are equal and, if they are not, prints an error
 * message to the standard error stream, including the expected and actual values, the current file
 * name, and the current line number, and then calls std::abort() to terminate the program.
 *
 * @param a The actual value to be compared.
 * @param b The expected value to be compared.
 */
#define NF_ASSERT_EQUAL(a, b) NF_ASSERT(a == b, "Expected " << b << ", got " << a)

/**
 * @def NF_DEBUG_ASSERT_EQUAL
 * @brief Macro for asserting that two values are equal and printing an error message if they are
 * not (only in debug mode).
 *
 * This macro is equivalent to NF_ASSERT_EQUAL in debug mode.
 *
 * @param a The actual value to be compared.
 * @param b The expected value to be compared.
 */
#define NF_DEBUG_ASSERT_EQUAL(a, b) NF_DEBUG_ASSERT(a == b, "Expected " << b << ", got " << a)

/**
 * @def NF_ASSERT_EQUAL_THROW
 * @brief Macro for asserting that two values are equal and throwing a NeoNException if they are
 * not.
 *
 * This macro checks that the two specified values are equal and, if they are not, constructs a
 * std::stringstream to concatenate the expected and actual values with the current file name and
 * current line number, and then throws a NeoNException with the resulting string as the error
 * message.
 *
 * @param a The actual value to be compared.
 * @param b The expected value to be compared.
 */
#define NF_ASSERT_EQUAL_THROW(a, b) NF_ASSERT_THROW(a == b, "Expected " << b << ", got " << a)

/**
 * @def NF_DEBUG_ASSERT_EQUAL_THROW
 * @brief Macro for asserting that two values are equal and throwing a NeoNException if they are
 * not (only in debug mode).
 *
 * This macro is equivalent to NF_ASSERT_EQUAL_THROW in debug mode.
 *
 * @param a The actual value to be compared.
 * @param b The expected value to be compared.
 */
#define NF_DEBUG_ASSERT_EQUAL_THROW(a, b)                                                          \
    NF_DEBUG_ASSERT_THROW(a == b, "Expected " << b << ", got " << a)
