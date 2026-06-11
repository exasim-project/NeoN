// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
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

    /**
     * @brief One-time patch initialisation, applied once per field before the first update().
     *
     * For patch data fixed for the lifetime of the field (e.g. constant mixed-BC coefficients).
     * Default: no-op.
     */
    virtual void set([[maybe_unused]] Field<ValueType>& domainVector) {}

    /**
     * @brief Per-iteration boundary update, applied on every correctBoundaryConditions() call.
     *
     * Default forwards to correctBoundaryCondition() so non-split BCs are unchanged.
     */
    virtual void update(Field<ValueType>& domainVector) { correctBoundaryCondition(domainVector); }

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

    virtual void set(Field<ValueType>& domainVector)
    {
        boundaryCorrectionStrategy_->set(domainVector);
    }

    virtual void update(Field<ValueType>& domainVector)
    {
        boundaryCorrectionStrategy_->update(domainVector);
    }


private:

    // NOTE needs full namespace to be not ambiguous
    std::unique_ptr<NeoN::finiteVolume::cellCentred::SurfaceBoundaryFactory<ValueType>>
        boundaryCorrectionStrategy_;
};


}
