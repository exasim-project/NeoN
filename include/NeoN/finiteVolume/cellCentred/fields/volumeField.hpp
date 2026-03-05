// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/database/database.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/domain.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/core/database/fieldDatabase.hpp"

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
class VolumeField : public DomainMixin<ValueType>, public FieldDatabaseMixin
{

public:

    using VectorValueType = ValueType;


    /**
     * @brief Constructor for a uninitialized VolumeField
     *
     * @param exec The executor
     * @param fieldName The name of the field
     * @param mesh The underlying mesh
     * @param boundaryConditions a vector of boundary conditions
     */
    VolumeField(
        const Executor& exec,
        std::string fieldName,
        const UnstructuredMesh& mesh,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
    );


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
    );

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
    );

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
    );

    VolumeField(const VolumeField& other);

    VolumeField<ValueType>& operator-=(const ValueType rhs);

    VolumeField<ValueType>& operator+=(const ValueType rhs);

    /**
     * @brief Corrects the boundary conditions of the surface field.
     *
     * This function applies the correctBoundaryConditions() method to each boundary condition in
     * the field.
     */
    void correctBoundaryConditions();

    std::vector<VolumeBoundary<ValueType>> boundaryConditions() const
    {
        return boundaryConditions_;
    }

private:

    std::vector<VolumeBoundary<ValueType>> boundaryConditions_; // The vector of boundary conditions
    std::optional<Database*> db_; // The optional pointer to the database
};

inline VolumeField<scalar> operator+(const VolumeField<scalar>& lhs, const VolumeField<scalar>& rhs)
{
    VolumeField<scalar> result(lhs);
    result.internalVector() += rhs.internalVector();
    result.boundaryData().value() += rhs.boundaryData().value();
    return result;
}

inline VolumeField<scalar> operator-(const VolumeField<scalar>& lhs, const VolumeField<scalar>& rhs)
{
    VolumeField<scalar> result(lhs);
    result.internalVector() -= rhs.internalVector();
    result.boundaryData().value() -= rhs.boundaryData().value();
    return result;
}

inline VolumeField<scalar> operator*(const VolumeField<scalar>& lhs, const VolumeField<scalar>& rhs)
{
    VolumeField<scalar> result(lhs);
    result.internalVector() *= rhs.internalVector();
    result.boundaryData().value() *= rhs.boundaryData().value();
    return result;
}

inline VolumeField<scalar> operator/(const VolumeField<scalar>& lhs, const VolumeField<scalar>& rhs)
{
    VolumeField<scalar> result(lhs);
    auto resultView = result.internalVector().view();
    auto rhsView = rhs.internalVector().view();
    parallelFor(
        result.internalVector().exec(),
        {0, result.internalVector().size()},
        NEON_LAMBDA(const localIdx i) { resultView[i] /= rhsView[i]; },
        "volumeFieldDiv"
    );
    auto bResultView = result.boundaryData().value().view();
    auto bRhsView = rhs.boundaryData().value().view();
    parallelFor(
        result.boundaryData().value().exec(),
        {0, result.boundaryData().value().size()},
        NEON_LAMBDA(const localIdx i) { bResultView[i] /= bRhsView[i]; },
        "volumeFieldDivBoundary"
    );
    return result;
}

} // namespace NeoN
