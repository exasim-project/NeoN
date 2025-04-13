// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary
{

template<typename ValueType>
class Calculated :
    public SurfaceBoundaryFactory<ValueType>::template Register<Calculated<ValueType>>
{
    using Base = SurfaceBoundaryFactory<ValueType>::template Register<Calculated<ValueType>>;

public:

    using CalculatedType = Calculated<ValueType>;

    Calculated(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID)
    {}

    virtual void correctBoundaryCondition([[maybe_unused]] Field<ValueType>& domainVector) override
    {}

    static std::string name() { return "calculated"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<SurfaceBoundaryFactory<ValueType>> clone() const override
    {
        return std::make_unique<Calculated>(*this);
    }
};
}
