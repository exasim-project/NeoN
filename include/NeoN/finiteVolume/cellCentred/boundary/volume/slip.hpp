// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/detail/slipSymmetry.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

// Slip is a frictionless-wall boundary condition: scalar => zero-gradient, vector => tangential
// projection + normal damping. It shares its implementation with Symmetry via
// detail::setSlipSymmetryValue; the two differ only in the registered name and in where they may be
// applied (slip on wall/patch types, symmetry on a symmetry-plane patch). The optional "implicit"
// key selects the normal-damping treatment.
template<typename ValueType>
class Slip : public VolumeBoundaryFactory<ValueType>::template Register<Slip<ValueType>>
{
    using Base = typename VolumeBoundaryFactory<ValueType>::template Register<Slip<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    using SlipType = Slip<ValueType>;

    Slip(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(
            mesh,
            dict,
            patchID,
            {.assignable = false,
             .fixesValue = false,
             .transformImplicit = detail::readTransformImplicit(dict)}
        ),
          mesh_(mesh), implicit_(detail::readTransformImplicit(dict))
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::setSlipSymmetryValue(
            domainVector, mesh_, this->range(), detail::normalDampingMode(implicit_)
        );
    }

    static std::string name() { return "slip"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Slip wall (scalar: zero-gradient; vector: tangential projection + normal damping).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Slip>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    bool implicit_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
