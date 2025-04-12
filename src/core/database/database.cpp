// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#include "NeoN/core/database/database.hpp"
#include "NeoN/core/database/collection.hpp"

namespace NeoN
{

Collection& Database::insert(const std::string& name, Collection&& col)
{
    return collections_.try_emplace(name, std::move(col)).first->second;
}

bool Database::contains(const std::string& name) const { return collections_.contains(name); }

Collection& Database::at(const std::string& name) { return collections_.at(name); }

const Collection& Database::at(const std::string& name) const { return collections_.at(name); }

std::size_t Database::size() const { return collections_.size(); }

bool Database::remove(const std::string& name) { return collections_.erase(name) > 0; }

} // namespace NeoN
