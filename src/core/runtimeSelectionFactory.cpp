// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <mutex>
#include <string>
#include <unordered_map>

#include "NeoN/core/runtimeSelectionFactory.hpp"

namespace NeoN::detail
{

void* sharedRegistryObject(const std::string& key, void* (*create)())
{
    // Leaked on purpose, see the declaration: no destructor runs at exit.
    static auto* objects = new std::unordered_map<std::string, void*>();
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    auto [it, inserted] = objects->try_emplace(key, nullptr);
    if (inserted)
    {
        it->second = create();
    }
    return it->second;
}

} // namespace NeoN::detail
