// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary
{


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
class Processor : public SurfaceBoundaryFactory<ValueType>::template Register<Processor<ValueType>>
{
    using Base = SurfaceBoundaryFactory<ValueType>::template Register<Processor<ValueType>>;

public:

    Processor(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition([[maybe_unused]] Field<ValueType>& domainVector) final
    {
        // detail::setProcBoundaryValue(domainVector, mesh_, this->range());
    }


    static std::string name() { return "processor"; }

    static std::string doc() { return "Do nothing on the boundary."; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<SurfaceBoundaryFactory<ValueType>> clone() const override
    {
        return std::make_unique<Processor>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

}
