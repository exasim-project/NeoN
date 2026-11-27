// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/core/visibility.hpp"

/* A factory base whose derived classes register from two shared objects: the test executable
 * and a plugin library compiled with -fvisibility=hidden -fvisibility-inlines-hidden, like the
 * Python bindings or a conda-forge build (#588).
 */
class SharedBase : public NeoN::RuntimeSelectionFactory<SharedBase, NeoN::Parameters<>>
{
public:

    SharedBase() = default;

    virtual std::string origin() const = 0;

    static std::string name() { return "SharedBase"; }
};

// Defined in the plugin and exported from it
NEON_EXPORT std::vector<std::string> pluginSeenEntries();
NEON_EXPORT std::unique_ptr<SharedBase> makePluginObject();
