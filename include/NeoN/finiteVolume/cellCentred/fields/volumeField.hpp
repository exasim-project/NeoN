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

    VolumeField<ValueType>& operator-=(const ValueType rhs);

    VolumeField<ValueType>& operator+=(const ValueType rhs);

    VolumeField<ValueType>& operator*=(const scalar rhs)
        requires requires(ValueType value, scalar rhsScalar) { value* rhsScalar; };

    [[nodiscard]] VolumeField<ValueType> operator*(const scalar rhs) const
        requires requires(ValueType value, scalar rhsScalar) { value* rhsScalar; };

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

    template<typename T = ValueType>
        requires std::is_same_v<T, Vec3>
    [[nodiscard]] VolumeField<scalar> component(const size_t componentIndex) const
    {
        return componentField(componentIndex, "component" + std::to_string(componentIndex));
    }

    template<typename T = ValueType>
        requires std::is_same_v<T, Vec3>
    [[nodiscard]] VolumeField<scalar> x() const
    {
        return componentField(0, "x");
    }

    template<typename T = ValueType>
        requires std::is_same_v<T, Vec3>
    [[nodiscard]] VolumeField<scalar> y() const
    {
        return componentField(1, "y");
    }

    template<typename T = ValueType>
        requires std::is_same_v<T, Vec3>
    [[nodiscard]] VolumeField<scalar> z() const
    {
        return componentField(2, "z");
    }

    template<typename T = ValueType>
        requires std::is_same_v<T, Vec3>
    [[nodiscard]] VolumeField<scalar>
    componentField(const size_t componentIndex, const std::string& suffix) const
    {
        if (componentIndex >= 3)
        {
            NF_ERROR_EXIT("VolumeField<Vec3>::component index out of range.");
        }

        VolumeField<scalar> result(
            this->exec(),
            this->name + "." + suffix,
            this->mesh(),
            createCalculatedBCs<VolumeBoundary<scalar>>(this->mesh())
        );

        auto sourceInternal = this->internalVector().view();
        parallelFor(
            result.internalVector(),
            NEON_LAMBDA(const localIdx i) { return sourceInternal[i][componentIndex]; }
        );

        auto sourceValue = this->boundaryData().value().view();
        parallelFor(
            result.boundaryData().value(),
            NEON_LAMBDA(const localIdx i) { return sourceValue[i][componentIndex]; }
        );

        auto sourceRefValue = this->boundaryData().refValue().view();
        parallelFor(
            result.boundaryData().refValue(),
            NEON_LAMBDA(const localIdx i) { return sourceRefValue[i][componentIndex]; }
        );

        auto sourceRefGrad = this->boundaryData().refGrad().view();
        parallelFor(
            result.boundaryData().refGrad(),
            NEON_LAMBDA(const localIdx i) { return sourceRefGrad[i][componentIndex]; }
        );

        result.boundaryData().valueFraction() = this->boundaryData().valueFraction();

        return result;
    }

private:

    std::vector<VolumeBoundary<ValueType>> boundaryConditions_; // The vector of boundary conditions
    std::optional<Database*> db_; // The optional pointer to the database
};

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

} // namespace NeoN
