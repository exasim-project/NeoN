// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoFOAM/core/runtimeSelectionFactory.hpp"                            // Register
#include "NeoFOAM/core/dictionary.hpp"                                         // Dictionary
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp" // VolumeBoundaryFactory
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"                      // UnstructuredMesh

namespace NeoFOAM::finiteVolume::cellCentred::volumeBoundary
{

template<typename ValueType>
class Empty : public VolumeBoundaryFactory<ValueType>::template Register<Empty<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<Empty<ValueType>>;

public:

    Empty(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID)
    {}

    virtual void correctBoundaryCondition([[maybe_unused]] DomainField<ValueType>& domainField
    ) final
    {}

    static std::string name() { return "empty"; }

    static std::string doc() { return "Do nothing on the boundary."; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Empty>(*this);
    }
};

}
