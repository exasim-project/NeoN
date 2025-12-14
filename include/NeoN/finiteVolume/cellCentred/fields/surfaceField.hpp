// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include "NeoN/core/database/database.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/domain.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoN/core/database/fieldDatabase.hpp"

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
class SurfaceField : public DomainMixin<ValueType>, public FieldDatabaseMixin
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
          FieldDatabaseMixin(), boundaryConditions_(boundaryConditions)
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
        : DomainMixin<ValueType>(exec, mesh, domainVector), FieldDatabaseMixin(),
          boundaryConditions_(boundaryConditions)
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
          FieldDatabaseMixin(), boundaryConditions_(boundaryConditions)
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
          FieldDatabaseMixin(db, std::move(dbKey), std::move(collectionName)),
          boundaryConditions_(boundaryConditions)
    {}

    /**
     * @brief Copy constructor for a surface field.
     *
     * @param other The surface field to copy.
     */
    SurfaceField(const SurfaceField& other)
        : DomainMixin<ValueType>(other), FieldDatabaseMixin(other),
          boundaryConditions_(other.boundaryConditions_)
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

    std::vector<SurfaceBoundary<ValueType>> boundaryConditions() const
    {
        return boundaryConditions_;
    }

private:

    std::vector<SurfaceBoundary<ValueType>>
        boundaryConditions_;      // The vector of boundary conditions
    std::optional<Database*> db_; // The optional pointer to the database
};

inline SurfaceField<scalar>
operator*(const SurfaceField<scalar>& lhs, const SurfaceField<scalar>& rhs)
{
    SurfaceField<scalar> result(lhs);
    result.internalVector() *= rhs.internalVector();
    result.boundaryData().value() *= rhs.boundaryData().value();
    return result;
}
inline SurfaceField<scalar>
operator+(const SurfaceField<scalar>& lhs, const SurfaceField<scalar>& rhs)
{
    SurfaceField<scalar> result(lhs);
    result.internalVector() += rhs.internalVector();
    result.boundaryData().value() += rhs.boundaryData().value();
    return result;
}

} // namespace NeoN
