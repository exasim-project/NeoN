// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include "NeoN/core/database/database.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/domain.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/**
 * @class SurfaceField
 * @brief Represents a surface field in a finite volume method.
 *
 * The SurfaceField class is a template class that represents a face-centered field in a finite
 * volume method. It inherits from the DomainMixin class and provides methods for correcting
 * boundary conditions.
 *
 * @tparam ValueType The value type of the field.
 */
template<typename ValueType>
class SurfaceField : public DomainMixin<ValueType>
{

public:

    using VectorValueType = ValueType;

    /**
     * @brief Constructor for a surfaceVector with a given name and mesh.
     *
     * @param exec The executor
     * @param fieldName The name of the field
     * @param mesh The underlying mesh
     * @param boundaryConditions a vector of boundary conditions
     */
    SurfaceField(
        const Executor& exec,
        std::string fieldName,
        const UnstructuredMesh& mesh,
        const std::vector<SurfaceBoundary<ValueType>>& boundaryConditions
    )
        : DomainMixin<ValueType>(
            exec,
            fieldName,
            mesh,
            Field<ValueType>(
                exec, mesh.nInternalFaces() + mesh.nBoundaryFaces(), mesh.boundaryMesh().offset()
            )
        ),
          boundaryConditions_(boundaryConditions)
    {}

    /* @brief Constructor for a surfaceVector with a given internal field
     *
     * @param exec The executor
     * @param mesh The underlying mesh
     * @param internalVector the underlying internal field
     * @param boundaryConditions a vector of boundary conditions
     */
    SurfaceField(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        const Field<ValueType>& domainVector,
        const std::vector<SurfaceBoundary<ValueType>>& boundaryConditions
    )
        : DomainMixin<ValueType>(exec, mesh, domainVector), boundaryConditions_(boundaryConditions)
    {}

    /* @brief Constructor for a surfaceVector with a given internal field
     *
     * @param exec The executor
     * @param mesh The underlying mesh
     * @param internalVector the underlying internal field
     * @param boundaryVectors the underlying boundary data fields
     * @param boundaryConditions a vector of boundary conditions
     */
    SurfaceField(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        const Vector<ValueType>& internalVector,
        const BoundaryData<ValueType>& boundaryVectors,
        const std::vector<SurfaceBoundary<ValueType>>& boundaryConditions
    )
        : DomainMixin<ValueType>(exec, mesh, {exec, mesh, internalVector, boundaryVectors}),
          boundaryConditions_(boundaryConditions)
    {}

    /* @brief Constructor for a surface field with a given internal field and database registration.
     *
     * Mirrors the VolumeField db-registering constructor.
     *
     * @param exec The executor
     * @param fieldName The name of the field
     * @param mesh The underlying mesh
     * @param domainVector the underlying domain field (internal + boundary laid out as Field)
     * @param boundaryConditions a vector of boundary conditions
     * @param db The database to register with
     * @param dbKey The key of the field in the database
     * @param collectionName The name of the field collection in the database
     */
    SurfaceField(
        const Executor& exec,
        std::string fieldName,
        const UnstructuredMesh& mesh,
        const Field<ValueType>& domainVector,
        const std::vector<SurfaceBoundary<ValueType>>& boundaryConditions,
        Database& db,
        std::string dbKey,
        std::string collectionName
    )
        : DomainMixin<ValueType>(exec, fieldName, mesh, domainVector),
          boundaryConditions_(boundaryConditions), db_(&db), key(std::move(dbKey)),
          fieldCollectionName(std::move(collectionName))
    {}

    /**
     * @brief Copy constructor for a surface field.
     *
     * @param other The surface field to copy.
     */
    SurfaceField(const SurfaceField& other)
        : DomainMixin<ValueType>(other), boundaryConditions_(other.boundaryConditions_),
          db_(other.db_), key(other.key), fieldCollectionName(other.fieldCollectionName)
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
     */
    bool hasDatabase() const { return db_.has_value(); }

    /**
     * @brief Retrieves the database.
     *
     * @throws std::runtime_error if the database is not set.
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
     * @brief Retrieves the database (const).
     *
     * @throws std::runtime_error if the database is not set.
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
     */
    bool registered() const { return key != "" && fieldCollectionName != "" && db_.has_value(); }

    std::vector<SurfaceBoundary<ValueType>> boundaryConditions() const
    {
        return boundaryConditions_;
    }

    std::string key;                 // The key of the field in the database
    std::string fieldCollectionName; // The name of the field collection in the database


private:

    std::vector<SurfaceBoundary<ValueType>>
        boundaryConditions_;      // The vector of boundary conditions
    std::optional<Database*> db_; // The optional pointer to the database
};


} // namespace NeoN
