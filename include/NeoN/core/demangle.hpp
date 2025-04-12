// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

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
    std::cerr << "Caught a bad_any_cast exception: \n"
              << "key " << key << "\n"
              << "requested type " << demangle(typeid(T).name()) << "\n"
              << "actual type " << demangle(data.at(key).type().name()) << "\n"
              << e.what() << std::endl;
}

}
