// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#include "NeoN/core/database/database.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/domain.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"

#include <vector>

namespace NeoN::finiteVolume::cellCentred
{

/**
 * @class VolumeField
 * @brief Represents a volume field in a finite volume method.
 *
 * The VolumeField class is a template class that represents a cell-centered field in a finite
 * volume method. It inherits from the DomainMixin class and provides methods for correcting
 * boundary conditions.
 *
 * @tparam ValueType The value type of the field.
 */
template<typename ValueType>
class VolumeField : public DomainMixin<ValueType>
{

public:

    using VectorValueType = ValueType;


    /**
     * @brief Constructor for a uninitialized VolumeField
     *
     * @param exec The executor
     * @param name The name of the field
     * @param mesh The underlying mesh
     * @param boundaryConditions a vector of boundary conditions
     */
    VolumeField(
        const Executor& exec,
        std::string name,
        const UnstructuredMesh& mesh,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
    )
        : DomainMixin<ValueType>(
            exec, name, mesh, Field<ValueType>(exec, mesh.nCells(), mesh.boundaryMesh().offset())
        ),
          key(""), fieldCollectionName(""), boundaryConditions_(boundaryConditions),
          db_(std::nullopt)
    {}


    /**
     * @brief Constructor for a VolumeField with a given internal field
     *
     * @param exec The executor
     * @param name The name of the field
     * @param mesh The underlying mesh
     * @param internalVector the underlying internal field
     * @param boundaryConditions a vector of boundary conditions
     */
    VolumeField(
        const Executor& exec,
        std::string name,
        const UnstructuredMesh& mesh,
        const Vector<ValueType>& internalVector,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
    )
        : DomainMixin<ValueType>(
            exec, name, mesh, Field<ValueType>(exec, internalVector, mesh.boundaryMesh().offset())
        ),
          key(""), fieldCollectionName(""), boundaryConditions_(boundaryConditions),
          db_(std::nullopt)
    {}

    /**
     * @brief Constructor for a VolumeField with a given internal and boundary field
     *
     * @param name The name of the field
     * @param mesh The underlying mesh
     * @param internalVector the underlying internal field
     * @param boundaryVectors the underlying boundary data fields
     * @param boundaryConditions a vector of boundary conditions
     */
    VolumeField(
        const Executor& exec,
        std::string name,
        const UnstructuredMesh& mesh,
        const Vector<ValueType>& internalVector,
        const BoundaryData<ValueType>& boundaryVectors,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
    )
        : DomainMixin<ValueType>(exec, name, mesh, internalVector, boundaryVectors), key(""),
          fieldCollectionName(""), boundaryConditions_(boundaryConditions), db_(std::nullopt)
    {}

    /**
     * @brief Constructor for a VolumeField with a given internal field and database
     *
     * @param exec The executor
     * @param fieldName The name of the field
     * @param mesh The underlying mesh
     * @param internalVector the underlying internal field
     * @param boundaryConditions a vector of boundary conditions
     * @param db The database
     * @param dbKey The key of the field in the database
     * @param collectionName The name of the field collection in the database
     */
    VolumeField(
        const Executor& exec,
        std::string fieldName,
        const UnstructuredMesh& mesh,
        const Field<ValueType>& domainVector,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions,
        Database& db,
        std::string dbKey,
        std::string collectionName
    )
        : DomainMixin<ValueType>(exec, fieldName, mesh, domainVector), key(dbKey),
          fieldCollectionName(collectionName), boundaryConditions_(boundaryConditions), db_(&db)
    {}

    VolumeField(const VolumeField& other)
        : DomainMixin<ValueType>(other), key(other.key),
          fieldCollectionName(other.fieldCollectionName),
          boundaryConditions_(other.boundaryConditions_), db_(other.db_)
    {}

    /**
     * @brief Corrects the boundary conditions of the surface field.
     *
     * This function applies the correctBoundaryConditions() method to each boundary condition in
     * the field.
     */
    void correctBoundaryConditions()
    {
        for (auto& boundaryCondition : boundaryConditions_)
        {
            boundaryCondition.correctBoundaryCondition(this->field_);
        }
    }

    /**
     * @brief Returns true if the field has a database, false otherwise.
     *
     * @return true if the field has a database, false otherwise.
     */
    bool hasDatabase() const { return db_.has_value(); }

    /**
     * @brief Retrieves the database.
     *
     * @return Database& A reference to the database.
     */
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

    /**
     * @brief Retrieves the database.
     *
     * @return const Database& A const reference to the database.
     */
    const Database& db() const
    {
        if (!db_.has_value())
        {
            throw std::runtime_error(
                "Database not set: make sure the field is registered in the database"
            );
        }
        return *db_.value();
    }

    /**
     * @brief Returns true if the field is registered in the database, false otherwise.
     *
     * @return true if the field is registered in the database, false otherwise.
     */
    bool registered() const { return key != "" && fieldCollectionName != "" && db_.has_value(); }

    std::vector<VolumeBoundary<ValueType>> boundaryConditions() const
    {
        return boundaryConditions_;
    }

    std::string key;                 // The key of the field in the database
    std::string fieldCollectionName; // The name of the field collection in the database

private:

    std::vector<VolumeBoundary<ValueType>> boundaryConditions_; // The vector of boundary conditions
    std::optional<Database*> db_; // The optional pointer to the database
};

} // namespace NeoN
