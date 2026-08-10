// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/database/database.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/domain.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/core/database/fieldDatabase.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

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

    void correctBoundaryConditions(const BoundaryContext& ctx);

    std::vector<VolumeBoundary<ValueType>> boundaryConditions() const
    {
        return boundaryConditions_;
    }

private:

    std::vector<VolumeBoundary<ValueType>> boundaryConditions_; // The vector of boundary conditions
    std::optional<Database*> db_; // The optional pointer to the database

    // Whether the one-time boundary set() pass has run for this field instance. Reset to false
    // for copies (default member init below) so a copied field re-runs set() on its first
    // correctBoundaryConditions() call.
    bool boundaryConditionsSet_ {false};
};

// Deliberately not called ``detail``: an inner ``detail`` here would shadow
// ``NeoN::detail`` for every unqualified lookup inside this namespace, which
// breaks divOperator/laplacianOperator's ``detail::RefHolder``.
namespace volumeFieldDetail
{

/** @brief in-place element-wise cross product, target = target ^ other */
inline void
crossInto(const Executor& exec, Vector<Vec3>& target, const Vector<Vec3>& other)
{
    auto out = target.view();
    auto rhs = other.view();
    parallelFor(
        exec,
        {0, out.size()},
        KOKKOS_LAMBDA(const localIdx i) { out[i] = out[i] ^ rhs[i]; },
        "vec3FieldCross"
    );
}

} // namespace volumeFieldDetail

/**
 * @brief Element-wise cross product of two Vec3 volume fields.
 *
 * Applied to the internal vector and to the boundary values alike, so the result
 * is usable wherever the operands were — matching the scalar SurfaceField
 * operators. Right-handed: ``cross(a, b)[i] == a[i] ^ b[i]``.
 */
inline VolumeField<Vec3> cross(const VolumeField<Vec3>& lhs, const VolumeField<Vec3>& rhs)
{
    VolumeField<Vec3> result(lhs);
    volumeFieldDetail::crossInto(result.exec(), result.internalVector(), rhs.internalVector());
    volumeFieldDetail::crossInto(result.exec(), result.boundaryData().value(), rhs.boundaryData().value());
    return result;
}

} // namespace NeoN
