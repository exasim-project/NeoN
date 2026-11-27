// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "sharedBase.hpp"

class PluginDerived : public SharedBase::Register<PluginDerived>
{
public:

    PluginDerived() = default;

    std::string origin() const override { return "plugin"; }

    static std::string name() { return "PluginDerived"; }

    static std::string doc() { return "registered by the plugin library"; }

    static std::string schema() { return "none"; }
};

std::vector<std::string> pluginSeenEntries() { return SharedBase::entries(); }

std::unique_ptr<SharedBase> makePluginObject() { return std::make_unique<PluginDerived>(); }
