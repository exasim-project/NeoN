// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/fields/domainField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/fields/boundaryFields.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/**
 * @class GeometricFieldMixin
 * @brief This class represents a mixin for a geometric field.
 *
 * The GeometricFieldMixin class provides a set of common operations and accessors for a geometric
 * field. It is designed to be used as a mixin in other classes that require geometric field
 * functionality.
 *
 * @tparam ValueType The value type of the field.
 */
template<typename ValueType>
class GeometricFieldMixin
{
public:


    typedef ValueType ElementType;

    /**
     * @brief Constructor for GeometricFieldMixin.
     *
     * @param exec The executor object.
     * @param fieldName The name of the field.
     * @param mesh The unstructured mesh object.
     * @param domainField The domain field object.
     */
    GeometricFieldMixin(
        const Executor& exec,
        std::string fieldName,
        const UnstructuredMesh& mesh,
        const DomainField<ValueType>& field
    )
        : name(fieldName), exec_(exec), mesh_(mesh), field_(field)
    {}

    /**
     * @brief Constructor for GeometricFieldMixin.
     *
     * @param exec The executor object.
     * @param fieldName The name of the corresponding field.
     * @param mesh The unstructured mesh object.
     * @param internalField The internal field object.
     * @param boundaryFields The boundary field object.
     */
    GeometricFieldMixin(
        const Executor& exec,
        std::string fieldName,
        const UnstructuredMesh& mesh,
        const Field<ValueType>& internalField,
        const BoundaryFields<ValueType>& boundaryFields
    )
        : name(fieldName), exec_(exec), mesh_(mesh), field_({exec, internalField, boundaryFields})
    {
        if (mesh.nCells() != internalField.size())
        {
            NF_ERROR_EXIT("Inconsistent size of mesh and internal field detected");
        }
    }

    /**
     * @brief Returns a const reference to the internal field.
     *
     * @return The const reference to the internal field.
     */
    const Field<ValueType>& internalField() const { return field_.internalField(); }

    /**
     * @brief Returns a reference to the internal field.
     *
     * @return The reference to the internal field.
     */
    Field<ValueType>& internalField() { return field_.internalField(); }

    /**
     * @brief Returns the size of the internal field
     *
     * @return The size of the internal field
     */
    size_t size() const { return field_.internalField().size(); }

    /**
     * @brief Returns a const reference to the boundary field.
     *
     * @return The const reference to the boundary field.
     */
    const BoundaryFields<ValueType>& boundaryField() const { return field_.boundaryField(); }

    /**
     * @brief Returns a reference to the boundary field.
     *
     * @return The reference to the boundary field.
     */
    BoundaryFields<ValueType>& boundaryField() { return field_.boundaryField(); }

    /**
     * @brief Returns a const reference to the executor object.
     *
     * @return The const reference to the executor object.
     */
    const Executor& exec() const { return exec_; }

    /**
     * @brief Returns a const reference to the unstructured mesh object.
     *
     * @return The const reference to the unstructured mesh object.
     */
    const UnstructuredMesh& mesh() const { return mesh_; }

    std::string name; // The name of the field

protected:

    Executor exec_;                // The executor object
    const UnstructuredMesh& mesh_; // The unstructured mesh object
    DomainField<ValueType> field_; // The domain field object
};

} // namespace NeoN
