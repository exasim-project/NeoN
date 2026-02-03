// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/database/database.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/domain.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
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

    /*    VolumeField<ValueType>& operator=(const VolumeField<ValueType>& rhs)
        {
            if (this != &rhs)
            {
                NF_DEBUG_ASSERT(&this->mesh_ == &rhs.mesh_, "VolumeField mesh mismatch.");
                this->name = rhs.name;
                this->field_ = rhs.field_;
                this->db_ = rhs.db_;
                this->key = rhs.key;
                this->fieldCollectionName = rhs.fieldCollectionName;
            }
            return *this;
        }
    */
    VolumeField<ValueType>& operator-=(const ValueType rhs);

    VolumeField<ValueType>& operator+=(const ValueType rhs);

    /*    VolumeField<ValueType>& operator*=(const scalar rhs)
            requires requires(ValueType value, scalar rhsScalar) { value* rhsScalar; };

        [[nodiscard]] VolumeField<ValueType> operator*(const scalar rhs) const
            requires requires(ValueType value, scalar rhsScalar) { value* rhsScalar; };
    */
    /**
     * @brief Corrects the boundary conditions of the surface field.
     *
     * This function applies the correctBoundaryConditions() method to each boundary condition in
     * the field.
     */
    void correctBoundaryConditions();

    void correctBoundaryConditions(const VolumeField<Vec3>& U, const VolumeField<scalar>& nu)
    {
        for (auto& boundaryCondition : boundaryConditions_)
        {
            boundaryCondition.correctBoundaryCondition(this->field_, U, nu);
        }
    }

    std::vector<VolumeBoundary<ValueType>> boundaryConditions() const
    {
        return boundaryConditions_;
    }

private:

    std::vector<VolumeBoundary<ValueType>> boundaryConditions_; // The vector of boundary conditions
    std::optional<Database*> db_; // The optional pointer to the database
};
/*
template<typename ValueType>
inline VolumeField<ValueType>
operator+(const VolumeField<ValueType>& lhs, const VolumeField<ValueType>& rhs)
{
    VolumeField<ValueType> result(lhs);
    add(result.internalVector(), rhs.internalVector());
    add(result.boundaryData().value(), rhs.boundaryData().value());
    return result;
}

template<typename ValueType>
inline VolumeField<ValueType>
operator-(const VolumeField<ValueType>& lhs, const VolumeField<ValueType>& rhs)
{
    VolumeField<ValueType> result(lhs);
    sub(result.internalVector(), rhs.internalVector());
    sub(result.boundaryData().value(), rhs.boundaryData().value());
    return result;
}

template<typename ValueType>
inline VolumeField<ValueType> operator*(scalar scale, const VolumeField<ValueType>& rhs)
    requires requires(ValueType value, scalar rhsScalar) { value* rhsScalar; }
{
    return rhs * scale;
}

template<typename ValueType>
inline VolumeField<ValueType> operator*(const VolumeField<ValueType>& lhs, scalar scale)
    requires requires(ValueType value, scalar rhsScalar) { value* rhsScalar; }
{
    return lhs * scale;
}

template<typename ValueType>
inline VolumeField<ValueType>
operator*(const VolumeField<ValueType>& lhs, const VolumeField<ValueType>& rhs)
    requires requires(ValueType value) { value* value; }
{
    VolumeField<ValueType> result(lhs);
    mul(result.internalVector(), rhs.internalVector());
    mul(result.boundaryData().value(), rhs.boundaryData().value());
    return result;
}

template<typename ValueType>
inline VolumeField<ValueType>
operator/(const VolumeField<ValueType>& lhs, const VolumeField<ValueType>& rhs)
    requires requires(ValueType value) { value / value; }
{
    VolumeField<ValueType> result(lhs);
    div(result.internalVector(), rhs.internalVector());
    div(result.boundaryData().value(), rhs.boundaryData().value());
    return result;
}

inline VolumeField<scalar> magSqr(const VolumeField<Vec3>& field)
{
    VolumeField<scalar> result(
        field.exec(),
        field.name + ".magSqr",
        field.mesh(),
        createCalculatedBCs<VolumeBoundary<scalar>>(field.mesh())
    );

    auto sourceInternal = field.internalVector().view();
    parallelFor(
        result.internalVector(),
        NEON_LAMBDA(const localIdx i) {
            const auto value = sourceInternal[i];
            return value[0] * value[0] + value[1] * value[1] + value[2] * value[2];
        }
    );
    return result;
}
*/
} // namespace NeoN
