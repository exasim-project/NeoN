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
class VolumeField;

class BoundaryContext;

/* collects attributes of a boundary for simple queries
 *
 */
struct BoundaryAttributes
{
    bool assignable; ///< whether values can be assigned to the boundary patch
    bool fixesValue; ///< whether the bc enforces a fixed field value (Dirichlet), not a
                     ///< gradient/Neumann condition NOTE this is somewhat redundant to
                     ///< valueFraction of boundaryData
};

template<typename ValueType>
class VolumeBoundaryFactory :
    public NeoN::RuntimeSelectionFactory<
        VolumeBoundaryFactory<ValueType>,
        Parameters<const UnstructuredMesh&, const Dictionary&, localIdx>>,
    public BoundaryPatchMixin
{
public:

    static std::string name() { return "VolumeBoundaryFactory"; }

    VolumeBoundaryFactory(
        const UnstructuredMesh& mesh,
        [[maybe_unused]] const Dictionary& dict,
        localIdx patchID,
        BoundaryAttributes attributes
    )
        : BoundaryPatchMixin(mesh, patchID), attributes_(attributes) {};

    virtual ~VolumeBoundaryFactory() = default;

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) = 0;

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector, const BoundaryContext&)
    {
        correctBoundaryCondition(domainVector);
    }

    virtual std::unique_ptr<VolumeBoundaryFactory> clone() const = 0;

    virtual std::string getName() const = 0;

    BoundaryAttributes attributes() const { return attributes_; }

protected:

    BoundaryAttributes attributes_; ///< The attributes of the patch
};


/**
 * @brief Represents a volume boundary field for a cell-centered finite volume method.
 *
 * @tparam ValueType The data type of the field.
 */
template<typename ValueType>
class VolumeBoundary : public BoundaryPatchMixin
{
public:

    VolumeBoundary(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : BoundaryPatchMixin(
            mesh.boundaryMesh().offset()[static_cast<size_t>(patchID)],
            mesh.boundaryMesh().offset()[static_cast<size_t>(patchID) + 1],
            patchID
        ),
          boundaryCorrectionStrategy_(VolumeBoundaryFactory<ValueType>::create(
              dict.get<std::string>("type"), mesh, dict, patchID
          ))
    {}

    VolumeBoundary(const VolumeBoundary& other)
        : BoundaryPatchMixin(other),
          boundaryCorrectionStrategy_(other.boundaryCorrectionStrategy_->clone())
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector)
    {
        boundaryCorrectionStrategy_->correctBoundaryCondition(domainVector);
    }

    virtual void
    correctBoundaryCondition(Field<ValueType>& domainVector, const BoundaryContext& ctx)
    {
        boundaryCorrectionStrategy_->correctBoundaryCondition(domainVector, ctx);
    }

    const BoundaryAttributes attributes() const
    {
        return boundaryCorrectionStrategy_->attributes();
    }

    const std::string name() const { return boundaryCorrectionStrategy_->getName(); }

private:

    // NOTE needs full namespace to be not ambiguous
    std::unique_ptr<NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<ValueType>>
        boundaryCorrectionStrategy_;
};

}
