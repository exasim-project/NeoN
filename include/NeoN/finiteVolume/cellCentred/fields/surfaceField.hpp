// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>
#include "NeoN/core/error.hpp"
#include "NeoN/core/database/database.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/vec3.hpp"
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
            Field<ValueType>(exec, mesh.nInternalFaces(), mesh.boundaryMesh().offset())
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
        Field<ValueType> domainVector,
        const std::vector<SurfaceBoundary<ValueType>>& boundaryConditions,
        Database& db,
        std::string dbKey,
        std::string collectionName
    )
        : DomainMixin<ValueType>(exec, fieldName, mesh, std::move(domainVector)),
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
     * @brief Move constructor. Without it, moving a SurfaceField would copy its data, since the
     * user-declared copy constructor suppresses the implicit move constructor.
     *
     * @param other The surface field to move from.
     */
    SurfaceField(SurfaceField&& other)
        : DomainMixin<ValueType>(std::move(other)), FieldDatabaseMixin(std::move(other)),
          boundaryConditions_(std::move(other.boundaryConditions_))
    {}

    /**
     * @brief Corrects the boundary conditions of the surface field.
     *
     * This function applies the correctBoundaryConditions() method to each boundary condition in
     * the field.
     */
    void correctBoundaryConditions()
    {
        // One-time set() pass: initialise constant patch data once per field instance (e.g. the
        // processor BC's unused mixed-BC coefficients).
        if (!boundaryConditionsSet_)
        {
            for (auto& boundaryCondition : boundaryConditions_)
            {
                boundaryCondition.set(this->field_);
            }
            boundaryConditionsSet_ = true;
        }
        // Per-iteration update(): proc patches post their (already face-valued) data without
        // draining, so all are in flight before any completes.
        for (auto& boundaryCondition : boundaryConditions_)
        {
            boundaryCondition.update(this->field_);
        }
        // Drain any processor-halo exchange once, after all proc patches have posted. See
        // VolumeField::correctBoundaryConditions for the rationale (batched post-then-wait).
        this->field_.boundaryData().waitAll();
    }

    std::vector<SurfaceBoundary<ValueType>> boundaryConditions() const
    {
        return boundaryConditions_;
    }

private:

    std::vector<SurfaceBoundary<ValueType>>
        boundaryConditions_;      // The vector of boundary conditions
    std::optional<Database*> db_; // The optional pointer to the database

    // Whether the one-time boundary set() pass has run for this field instance. Default-init to
    // false so copies re-run set() on their first correctBoundaryConditions() call.
    bool boundaryConditionsSet_ {false};
};

namespace surfaceFieldOps
{

/**
 * @brief Throws unless both surface fields can be combined element by element.
 *
 * The operators below launch kernels over the size of one operand and read the other, so a
 * shorter or differently placed operand would be read out of bounds. The checks are O(1) and
 * always on; the Vector operators' own checks are debug-only and do not cover the Vec3-by-scalar
 * kernel at all.
 *
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @param op The operation, for the error message.
 */
template<typename LhsType, typename RhsType>
inline void validateSurfaceFieldOperands(
    const SurfaceField<LhsType>& lhs, const SurfaceField<RhsType>& rhs, const char* op
)
{
    NF_ASSERT_THROW(
        lhs.exec() == rhs.exec(),
        "SurfaceField '" << lhs.name << "' " << op << " '" << rhs.name << "': executors differ"
    );
    NF_ASSERT_THROW(
        lhs.internalVector().size() == rhs.internalVector().size(),
        "SurfaceField '" << lhs.name << "' " << op << " '" << rhs.name
                         << "': " << lhs.internalVector().size() << " and "
                         << rhs.internalVector().size() << " internal faces"
    );
    NF_ASSERT_THROW(
        lhs.boundaryData().value().size() == rhs.boundaryData().value().size(),
        "SurfaceField '" << lhs.name << "' " << op << " '" << rhs.name
                         << "': " << lhs.boundaryData().value().size() << " and "
                         << rhs.boundaryData().value().size() << " boundary faces"
    );
}

/**
 * @brief The result of an arithmetic expression: a new, unregistered field named like `like`,
 * that starts from the internal and boundary values of `like` and carries 'calculated' boundary
 * conditions.
 *
 * Copying an operand would also copy its boundary conditions. A fixedValue (or any other
 * value-setting) patch of the operand would then overwrite the boundary values the expression
 * has just computed at the next correctBoundaryConditions(), and the result would inherit the
 * operand's database registration. OpenFOAM's operators likewise return fields with calculated
 * patches. (The calculated patches are built here rather than with createCalculatedBCs because
 * boundary.hpp includes this header.)
 *
 * @param like The operand whose name, mesh and values the result starts from.
 */
template<typename ValueType>
inline SurfaceField<ValueType> arithmeticResult(const SurfaceField<ValueType>& like)
{
    const auto& mesh = like.mesh();
    std::vector<SurfaceBoundary<ValueType>> bcs;
    bcs.reserve(mesh.nBoundaries());
    const auto nBoundaries = toLocalIdx(mesh.nBoundaries());
    for (localIdx patchID = 0; patchID < nBoundaries; ++patchID)
    {
        Dictionary patchDict({{"type", std::string("calculated")}});
        bcs.emplace_back(mesh, patchDict, patchID);
    }
    SurfaceField<ValueType> result(like.exec(), like.name, mesh, bcs);
    result.internalVector() = like.internalVector();
    result.boundaryData().value() = like.boundaryData().value();
    return result;
}

} // namespace surfaceFieldOps

inline SurfaceField<scalar>
operator*(const SurfaceField<scalar>& lhs, const SurfaceField<scalar>& rhs)
{
    surfaceFieldOps::validateSurfaceFieldOperands(lhs, rhs, "*");
    auto result = surfaceFieldOps::arithmeticResult(lhs);
    result.internalVector() *= rhs.internalVector();
    result.boundaryData().value() *= rhs.boundaryData().value();
    return result;
}

inline SurfaceField<scalar>
operator+(const SurfaceField<scalar>& lhs, const SurfaceField<scalar>& rhs)
{
    surfaceFieldOps::validateSurfaceFieldOperands(lhs, rhs, "+");
    auto result = surfaceFieldOps::arithmeticResult(lhs);
    result.internalVector() += rhs.internalVector();
    result.boundaryData().value() += rhs.boundaryData().value();
    return result;
}

inline SurfaceField<scalar>
operator-(const SurfaceField<scalar>& lhs, const SurfaceField<scalar>& rhs)
{
    surfaceFieldOps::validateSurfaceFieldOperands(lhs, rhs, "-");
    auto result = surfaceFieldOps::arithmeticResult(lhs);
    result.internalVector() -= rhs.internalVector();
    result.boundaryData().value() -= rhs.boundaryData().value();
    return result;
}

inline SurfaceField<scalar> operator*(scalar s, const SurfaceField<scalar>& fld)
{
    auto result = surfaceFieldOps::arithmeticResult(fld);
    result.internalVector() *= s;
    result.boundaryData().value() *= s;
    return result;
}

inline void surfaceFieldScaleVec3ByScalar(Vector<Vec3>& out, const Vector<scalar>& coeff)
{
    NF_ASSERT_THROW(
        out.size() == coeff.size(),
        "surfaceFieldScaleVec3ByScalar: " << out.size() << " values and " << coeff.size()
                                          << " coefficients"
    );
    NF_ASSERT_THROW(out.exec() == coeff.exec(), "surfaceFieldScaleVec3ByScalar: executors differ");
    auto [outV, coeffV] = views(out, coeff);
    parallelFor(
        out.exec(),
        {0, out.size()},
        NEON_LAMBDA(const localIdx i) { outV[i] = outV[i] * coeffV[i]; },
        "surfaceField::scaleVec3ByScalar"
    );
}

inline SurfaceField<Vec3> operator*(const SurfaceField<scalar>& s, const SurfaceField<Vec3>& v)
{
    surfaceFieldOps::validateSurfaceFieldOperands(s, v, "*");
    auto result = surfaceFieldOps::arithmeticResult(v);
    surfaceFieldScaleVec3ByScalar(result.internalVector(), s.internalVector());
    surfaceFieldScaleVec3ByScalar(result.boundaryData().value(), s.boundaryData().value());
    return result;
}

inline SurfaceField<Vec3> operator*(const SurfaceField<Vec3>& v, const SurfaceField<scalar>& s)
{
    return s * v;
}

inline SurfaceField<Vec3> operator+(const SurfaceField<Vec3>& lhs, const SurfaceField<Vec3>& rhs)
{
    surfaceFieldOps::validateSurfaceFieldOperands(lhs, rhs, "+");
    auto result = surfaceFieldOps::arithmeticResult(lhs);
    result.internalVector() += rhs.internalVector();
    result.boundaryData().value() += rhs.boundaryData().value();
    return result;
}

inline SurfaceField<Vec3> operator-(const SurfaceField<Vec3>& lhs, const SurfaceField<Vec3>& rhs)
{
    surfaceFieldOps::validateSurfaceFieldOperands(lhs, rhs, "-");
    auto result = surfaceFieldOps::arithmeticResult(lhs);
    result.internalVector() -= rhs.internalVector();
    result.boundaryData().value() -= rhs.boundaryData().value();
    return result;
}

} // namespace NeoN
