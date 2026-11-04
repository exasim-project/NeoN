// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once
// TODO For WIN builds, needs to be ifdef'ed out.
#ifdef __GNUC__
#include <cxxabi.h>
#endif
#include <string>
#include <any>
#include <iostream>

namespace NeoN
{

std::string demangle(const char* mangledName);

template<typename T, typename Container, typename Key>
void logBadAnyCast(const std::bad_any_cast& e, const Key& key, const Container& data)
{
    std::cerr << "bad_any_cast: key='" << key << "', requested='" << demangle(typeid(T).name())
              << "', actual='" << demangle(data.at(key).type().name()) << "': " << e.what() << "\n";
}

}
