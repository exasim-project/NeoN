// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/core/vector.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/fields/boundaryData.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/**
 * @class DomainMixin
 * @brief This class represents a mixin for a geometric field.
 *
 * The DomainMixin class provides a set of common operations and accessors for a geometric
 * field. It is designed to be used as a mixin in other classes that require geometric field
 * functionality.
 *
 * @tparam ValueType The value type of the field.
 */
template<typename ValueType>
class DomainMixin
{
public:


    typedef ValueType ElementType;

    /**
     * @brief Constructor for DomainMixin.
     *
     * @param exec The executor object.
     * @param fieldName The name of the field.
     * @param mesh The unstructured mesh object.
     * @param domainVector The domain field object.
     */
    DomainMixin(
        const Executor& exec,
        std::string fieldName,
        const UnstructuredMesh& mesh,
        const Field<ValueType>& field
    )
        : name(fieldName), exec_(exec), mesh_(mesh), field_(field)
    {}

    /**
     * @brief Constructor for DomainMixin.
     *
     * @param exec The executor object.
     * @param fieldName The name of the corresponding field.
     * @param mesh The unstructured mesh object.
     * @param internalVector The internal field object.
     * @param boundaryVectors The boundary field object.
     */
    DomainMixin(
        const Executor& exec,
        std::string fieldName,
        const UnstructuredMesh& mesh,
        const Vector<ValueType>& internalVector,
        const BoundaryData<ValueType>& boundaryVectors
    )
        : name(fieldName), exec_(exec), mesh_(mesh), field_({exec, internalVector, boundaryVectors})
    {
        if (mesh.nCells() != internalVector.size())
        {
            NF_ERROR_EXIT("Inconsistent size of mesh and internal field detected");
        }
    }

    /**
     * @brief Returns a const reference to the internal field.
     *
     * @return The const reference to the internal field.
     */
    const Vector<ValueType>& internalVector() const { return field_.internalVector(); }

    /**
     * @brief Returns a reference to the internal field.
     *
     * @return The reference to the internal field.
     */
    Vector<ValueType>& internalVector() { return field_.internalVector(); }

    /**
     * @brief Returns the size of the internal field
     *
     * @return The size of the internal field
     */
    localIdx size() const { return field_.internalVector().size(); }

    /**
     * @brief Returns a const reference to the boundary field.
     *
     * @return The const reference to the boundary field.
     */
    const BoundaryData<ValueType>& boundaryData() const { return field_.boundaryData(); }

    /**
     * @brief Returns a reference to the boundary field.
     *
     * @return The reference to the boundary field.
     */
    BoundaryData<ValueType>& boundaryData() { return field_.boundaryData(); }

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
    Field<ValueType> field_;       // The domain field object
};

} // namespace NeoN
