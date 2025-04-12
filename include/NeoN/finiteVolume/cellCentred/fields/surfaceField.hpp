// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#include <vector>

#include "NeoN/finiteVolume/cellCentred/fields/geometricField.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/**
 * @class SurfaceField
 * @brief Represents a surface field in a finite volume method.
 *
 * The SurfaceField class is a template class that represents a face-centered field in a finite
 * volume method. It inherits from the GeometricFieldMixin class and provides methods for correcting
 * boundary conditions.
 *
 * @tparam ValueType The value type of the field.
 */
template<typename ValueType>
class SurfaceField : public GeometricFieldMixin<ValueType>
{

public:

    /**
     * @brief Constructor for a surfaceField with a given name and mesh.
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
        : GeometricFieldMixin<ValueType>(
            exec,
            fieldName,
            mesh,
            DomainField<ValueType>(
                exec, mesh.nInternalFaces() + mesh.nBoundaryFaces(), mesh.boundaryMesh().offset()
            )
        ),
          boundaryConditions_(boundaryConditions)
    {}

    /* @brief Constructor for a surfaceField with a given internal field
     *
     * @param exec The executor
     * @param mesh The underlying mesh
     * @param internalField the underlying internal field
     * @param boundaryConditions a vector of boundary conditions
     */
    SurfaceField(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        const DomainField<ValueType>& domainField,
        const std::vector<SurfaceBoundary<ValueType>>& boundaryConditions
    )
        : GeometricFieldMixin<ValueType>(exec, mesh, domainField),
          boundaryConditions_(boundaryConditions)
    {}

    /* @brief Constructor for a surfaceField with a given internal field
     *
     * @param exec The executor
     * @param mesh The underlying mesh
     * @param internalField the underlying internal field
     * @param boundaryConditions a vector of boundary conditions
     */
    SurfaceField(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        const Field<ValueType>& internalField,
        const BoundaryFields<ValueType>& boundaryFields,
        const std::vector<SurfaceBoundary<ValueType>>& boundaryConditions
    )
        : GeometricFieldMixin<ValueType>(exec, mesh, {exec, mesh, internalField, boundaryFields}),
          boundaryConditions_(boundaryConditions)
    {}

    /**
     * @brief Copy constructor for a surface field.
     *
     * @param other The surface field to copy.
     */
    SurfaceField(const SurfaceField& other)
        : GeometricFieldMixin<ValueType>(other), boundaryConditions_(other.boundaryConditions_)
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


private:

    std::vector<SurfaceBoundary<ValueType>>
        boundaryConditions_; // The vector of boundary conditions
};


} // namespace NeoN
