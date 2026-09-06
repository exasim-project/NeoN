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

// Symmetry is a symmetry-plane boundary condition: scalar => zero-gradient, vector => tangential
// projection + normal damping. It shares its implementation with Slip via
// detail::setSlipSymmetryValue; they differ only in the registered name and in where they may be
// applied. The optional "implicit" key selects the normal-damping treatment.
template<typename ValueType>
class Symmetry : public VolumeBoundaryFactory<ValueType>::template Register<Symmetry<ValueType>>
{
    using Base = typename VolumeBoundaryFactory<ValueType>::template Register<Symmetry<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    using SymmetryType = Symmetry<ValueType>;

    Symmetry(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
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

    static std::string name() { return "symmetry"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Symmetry plane (scalar: zero-gradient; vector: tangential projection + normal "
               "damping).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Symmetry>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    bool implicit_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
