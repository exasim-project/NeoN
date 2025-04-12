// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoFOAM::finiteVolume::cellCentred::surfaceBoundary
{

template<typename ValueType>
class Empty : public SurfaceBoundaryFactory<ValueType>::template Register<Empty<ValueType>>
{
    using Base = SurfaceBoundaryFactory<ValueType>::template Register<Empty<ValueType>>;

public:

    Empty(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID)
    {}

    virtual void correctBoundaryCondition([[maybe_unused]] DomainField<ValueType>& domainField
    ) override
    {}

    static std::string name() { return "empty"; }

    static std::string doc() { return "Do nothing on the boundary."; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<SurfaceBoundaryFactory<ValueType>> clone() const override
    {
        return std::make_unique<Empty>(*this);
    }
};

}
