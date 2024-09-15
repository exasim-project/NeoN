// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>

#include "NeoFOAM/finiteVolume/cellCentred/fields/geometricField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
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

    SurfaceField(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        const std::vector<SurfaceBoundary<ValueType>>& boundaryConditions
    )
        : GeometricFieldMixin<ValueType>(
            exec,
            mesh,
            DomainField<ValueType>(
                exec,
                mesh.nInternalFaces() + mesh.nBoundaryFaces(),
                mesh.nBoundaryFaces(),
                mesh.nBoundaries()
            )
        ),
          boundaryConditions_(boundaryConditions)
    {}

    SurfaceField(
        const Executor& exec,
        std::string name,
        const UnstructuredMesh& mesh,
        SolutionFields& solField,
        const std::vector<SurfaceBoundary<ValueType>>& boundaryConditions
    )
        : GeometricFieldMixin<ValueType>(
            exec,
            name,
            mesh,
            DomainField<ValueType>(
                exec,
                mesh.nInternalFaces() + mesh.nBoundaryFaces(),
                mesh.nBoundaryFaces(),
                mesh.nBoundaries()
            ),
            solField
        ),
          boundaryConditions_(boundaryConditions)
    {}


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


} // namespace NeoFOAM
