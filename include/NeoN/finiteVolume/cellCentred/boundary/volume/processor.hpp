// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/mpi/operators.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

// TODO move to source file
namespace detail
{
// NOTE test with zero gradient first
// FIXME TODO exchange values on boundaries with neighbour rank
template<typename ValueType>
void setProcBoundaryValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto iVector = domainVector.internalVector().view();

    auto [refGradient, value, valueFraction, refValue, faceCells, deltaCoeffs] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceCells(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            refGradient[i] = zero<ValueType>();
            value[i] = iVector[faceCells[i]];
            valueFraction[i] = 0.0;          // only use refGrad
            refValue[i] = zero<ValueType>(); // not used
        },
        "setProcBoundaryValue"
    );
}
}

template<typename ValueType>
class Processor : public VolumeBoundaryFactory<ValueType>::template Register<Processor<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<Processor<ValueType>>;

public:

    using ProcessorType = Processor<ValueType>;

    Processor(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true}), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition([[maybe_unused]] Field<ValueType>& domainVector) final
    {
        detail::setProcBoundaryValue(domainVector, mesh_, this->range());
    }

    static std::string name() { return "processor"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Processor>(*this);
    }

    virtual std::string getName() const { return name(); }


private:

    const UnstructuredMesh& mesh_;
};
}
