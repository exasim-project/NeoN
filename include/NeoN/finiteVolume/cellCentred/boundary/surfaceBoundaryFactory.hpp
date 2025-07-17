// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryPatchMixin.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
class SurfaceBoundaryFactory :
    public NeoN::RuntimeSelectionFactory<
        SurfaceBoundaryFactory<ValueType>,
        Parameters<const UnstructuredMesh&, const Dictionary&, localIdx>>,
    public BoundaryPatchMixin
{
public:

    static std::string name() { return "SurfaceBoundaryFactory"; }

    SurfaceBoundaryFactory(
        const UnstructuredMesh& mesh, [[maybe_unused]] const Dictionary&, localIdx patchID
    )
        : BoundaryPatchMixin(mesh, patchID) {};

    virtual void correctBoundaryCondition(Field<ValueType>& field) = 0;

    virtual std::unique_ptr<SurfaceBoundaryFactory> clone() const = 0;
};


/**
 * @brief Represents a surface boundary field for a cell-centered finite volume method.
 *
 * @tparam ValueType The data type of the field.
 */
template<typename ValueType>
class SurfaceBoundary : public BoundaryPatchMixin
{
public:

    SurfaceBoundary(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : BoundaryPatchMixin(
            mesh.boundaryMesh().offset()[static_cast<size_t>(patchID)],
            mesh.boundaryMesh().offset()[static_cast<size_t>(patchID) + 1],
            patchID
        ),
          boundaryCorrectionStrategy_(SurfaceBoundaryFactory<ValueType>::create(
              dict.get<std::string>("type"), mesh, dict, patchID
          ))
    {}

    SurfaceBoundary(const SurfaceBoundary& other)
        : BoundaryPatchMixin(other),
          boundaryCorrectionStrategy_(other.boundaryCorrectionStrategy_->clone())
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector)
    {
        boundaryCorrectionStrategy_->correctBoundaryCondition(domainVector);
    }


private:

    // NOTE needs full namespace to be not ambiguous
    std::unique_ptr<NeoN::finiteVolume::cellCentred::SurfaceBoundaryFactory<ValueType>>
        boundaryCorrectionStrategy_;
};


}
