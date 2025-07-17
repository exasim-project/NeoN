// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#ifdef __GNUC__
#include <cxxabi.h> // for __cxa_demangle
#endif
#include <stdlib.h> // for free

#include "NeoN/core/demangle.hpp"


std::string NeoN::demangle(const char* name)
{
#ifdef _MSC_VER
    return name; // For MSVC, return the name directly.
#elif defined(__GNUC__)
    int status;
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0)
    {
        std::string result(demangled);
        free(demangled);
        return result;
    }
    else
    {
        return name;
    }
#endif
}
