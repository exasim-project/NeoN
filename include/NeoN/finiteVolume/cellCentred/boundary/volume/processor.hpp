// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{
template<typename ValueType>
void setProcBoundaryValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
);

extern template void setProcBoundaryValue<
    scalar>(Field<scalar>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);
extern template void
setProcBoundaryValue<Vec3>(Field<Vec3>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);
}

template<typename ValueType>
class Processor : public VolumeBoundaryFactory<ValueType>::template Register<Processor<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<Processor<ValueType>>;

public:

    using ProcessorType = Processor<ValueType>;

    Processor(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = false}), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition([[maybe_unused]] Field<ValueType>& domainVector) final
    {
        detail::setProcBoundaryValue(domainVector, mesh_, this->range());
    }

    static std::string name() { return "processor"; }

    static std::string doc()
    {
        return "Set processor boundary values from the corresponding processor-neighbour data.";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Processor>(*this);
    }

    std::string getName() const override { return name(); }


private:

    const UnstructuredMesh& mesh_;
};
}
