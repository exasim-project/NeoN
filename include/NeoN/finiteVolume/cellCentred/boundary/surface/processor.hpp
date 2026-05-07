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
// Surface processor BC: stage the locally-computed face value into
// boundaryData.value() so that the subsequent MPI exchange in
// SurfaceField::correctBoundaryConditions() has well-defined sender data.
//
// Indexing: SurfaceField::internalVector() spans [0, nTotalFaces) in the
// compressed face order. For processor faces, compressed face index is
// `nInternalFaces + bcfaceI` where bcfaceI is the boundary face index
// (the index space `range` is expressed in).
template<typename ValueType>
void setProcBoundaryValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto iVector = domainVector.internalVector().view();
    auto value = domainVector.boundaryData().value().view();
    const auto nInternalFaces = mesh.nInternalFaces();

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) { value[i] = iVector[nInternalFaces + i]; },
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

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::setProcBoundaryValue(domainVector, mesh_, this->range());
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
