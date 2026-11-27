// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <algorithm>

#include "catch2_common.hpp"

#include "sharedBase.hpp"

class ExeDerived : public SharedBase::Register<ExeDerived>
{
public:

    ExeDerived() = default;

    std::string origin() const override { return "executable"; }

    static std::string name() { return "ExeDerived"; }

    static std::string doc() { return "registered by the test executable"; }

    static std::string schema() { return "none"; }
};

namespace
{
bool contains(const std::vector<std::string>& entries, const std::string& name)
{
    return std::find(entries.begin(), entries.end(), name) != entries.end();
}
} // namespace

TEST_CASE("Runtime-selection table is shared across shared objects (#588)")
{
    // Instantiate the registrations of both sides
    REQUIRE(ExeDerived().origin() == "executable");
    REQUIRE(makePluginObject()->origin() == "plugin");

    SECTION("the executable sees the plugin's registration")
    {
        REQUIRE(contains(SharedBase::entries(), "PluginDerived"));
        REQUIRE(SharedBase::create("PluginDerived")->origin() == "plugin");
        REQUIRE(SharedBase::doc("PluginDerived") == "registered by the plugin library");
    }

    SECTION("the plugin sees the executable's registration")
    {
        const auto seen = pluginSeenEntries();
        REQUIRE(contains(seen, "ExeDerived"));
        REQUIRE(contains(seen, "PluginDerived"));
        REQUIRE(seen.size() == SharedBase::size());
    }
}
