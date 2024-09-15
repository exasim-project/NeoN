// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>
#include <optional>
#include <functional>


#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/fields/domainField.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"


namespace NeoFOAM::finiteVolume::cellCentred
{

// // forward declaration
class SolutionFields;

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

    /**
     * @brief Constructor for GeometricFieldMixin.
     *
     * @param exec The executor object.
     * @param mesh The unstructured mesh object.
     * @param field The domain field object.
     */
    GeometricFieldMixin(
        const Executor& exec, const UnstructuredMesh& mesh, const DomainField<ValueType>& field
    )
        : exec_(exec), mesh_(mesh), field_(field)
    {}

    GeometricFieldMixin(
        const Executor& exec,
        std::string fieldName,
        const UnstructuredMesh& mesh,
        const DomainField<ValueType>& field,
        SolutionFields& solField
    )
        : exec_(exec), name(fieldName), mesh_(mesh), field_(field), solField_(solField)
    {}

        GeometricFieldMixin(
        const GeometricFieldMixin& geomFieldMixin,
        SolutionFields& solField
    )
        : exec_(geomFieldMixin.exec()), name(geomFieldMixin.name), mesh_(geomFieldMixin.mesh()), field_(geomFieldMixin.internalField()), solField_(solField)
    {}


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

    /**
     * @brief Returns a const reference to the solution field object.
     *
     * @return The const reference to the solution field object.
    */
    const SolutionFields& solField() const { return solField_.value(); }

    /**
     * @brief Returns a reference to the solution field object.
     *
     * @return The reference to the solution field object.
    */
    SolutionFields& solField() { return solField_.value(); }

    bool hasSolField() const { return solField_.has_value(); }

    void setSolField(SolutionFields& solField) { solField_ = solField; }

protected:

    Executor exec_;                // The executor object
    const UnstructuredMesh& mesh_; // The unstructured mesh object
    DomainField<ValueType> field_; // The domain field object
    std::optional<std::reference_wrapper<SolutionFields>> solField_; // The solution field object
};

} // namespace NeoFOAM
