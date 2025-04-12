// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/fields/domainField.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryPatchMixin.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
class SurfaceBoundaryFactory :
    public NeoN::RuntimeSelectionFactory<
        SurfaceBoundaryFactory<ValueType>,
        Parameters<const UnstructuredMesh&, const Dictionary&, size_t>>,
    public BoundaryPatchMixin
{
public:

    static std::string name() { return "SurfaceBoundaryFactory"; }

    SurfaceBoundaryFactory(
        const UnstructuredMesh& mesh, [[maybe_unused]] const Dictionary&, size_t patchID
    )
        : BoundaryPatchMixin(mesh, patchID) {};

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) = 0;

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

    SurfaceBoundary(const UnstructuredMesh& mesh, const Dictionary& dict, size_t patchID)
        : BoundaryPatchMixin(
            static_cast<label>(mesh.boundaryMesh().offset()[patchID]),
            static_cast<label>(mesh.boundaryMesh().offset()[patchID + 1]),
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

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField)
    {
        boundaryCorrectionStrategy_->correctBoundaryCondition(domainField);
    }


private:

    // NOTE needs full namespace to be not ambiguous
    std::unique_ptr<NeoN::finiteVolume::cellCentred::SurfaceBoundaryFactory<ValueType>>
        boundaryCorrectionStrategy_;
};


}
