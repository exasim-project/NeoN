// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

/**
 * @def NF_DEBUG_MESSAGING
 * @brief Macro to enable debug messaging.
 *        It is enabled if NF_DEBUG or NF_DEBUG_INFO is defined.
 */
#if defined(NF_DEBUG) || defined(NF_DEBUG_INFO)
#define NF_DEBUG_MESSAGING
#endif

/**
 * @def NF_INFO(message)
 * @brief Prints the given message to the standard output stream.
 * @param message The message to be printed.
 */
#define NF_INFO(message) std::cout << message << std::endl

/**
 * @def NF_DINFO(message)
 * @brief Prints the given debug message to the standard output stream if
 * NF_DEBUG_MESSAGING is enabled.
 * @param message The debug message to be printed.
 */
#ifdef NF_DEBUG_MESSAGING
#define NF_DINFO(message)                                                                          \
    std::cout << std::endl << "[NF][" << __FILE__ << "::" << __LINE__ << "]: ";                    \
    NF_INFO(message)
#else
#define NF_DINFO(message) ((void)0)
#endif

/**
 * @def PING()
 * @brief Prints a very clear message that the code reaches a certain file and line.
 */
#ifdef NF_DEBUG
#define NF_PING() NF_DINFO("PING")
#endif
