// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>
#include <stdexcept>

#include "NeoN/core/database/database.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/**
 * @brief Mixin for database registration for fields (volume/surface).
 *
 * Provides:
 *   - optional Database* storage
 *   - key, fieldCollectionName
 *   - hasDatabase(), db(), registered()
 */
class FieldDatabaseMixin
{
public:

    FieldDatabaseMixin() = default;

    FieldDatabaseMixin(Database& db, std::string keyIn, std::string collectionName)
        : key(std::move(keyIn)), fieldCollectionName(std::move(collectionName)), db_(&db)
    {}

    bool hasDatabase() const { return db_.has_value(); }

    Database& db()
    {
        if (!db_.has_value())
        {
            throw std::runtime_error {
                "Database not set: make sure the field is registered in the database"
            };
        }
        return *db_.value();
    }

    const Database& db() const
    {
        if (!db_.has_value())
        {
            throw std::runtime_error {
                "Database not set: make sure the field is registered in the database"
            };
        }
        return *db_.value();
    }

    /* @brief tests whether the given object has been registered in a database */
    bool registered() const
    {
        return !key.empty() && !fieldCollectionName.empty() && db_.has_value();
    }

    // Keep names to match your existing API
    std::string key;                 // key in DB
    std::string fieldCollectionName; // collection name in DB

protected:

    std::optional<Database*> db_;
};

} // namespace NeoN::finiteVolume::cellCentred
