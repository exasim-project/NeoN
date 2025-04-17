// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

// TODO move to source file
namespace detail
{
// Without this function the compiler warns that calling a __host__ function
// from a __device__ function is not allowed
// NOTE: patchID was removed since it was unused
// I guess it was replaced by range
template<typename ValueType>
void extrapolateValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto iVector = domainVector.internalVector().view();

    auto [refGradient, value, valueFraction, refValue, faceCells] = spans(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceCells()
    );


    NeoN::parallelFor(
        domainVector.exec(),
        range,
        KOKKOS_LAMBDA(const localIdx i) {
            // operator / is not defined for all ValueTypes
            ValueType internalCellValue = iVector[faceCells[i]];
            value[i] = internalCellValue;
            valueFraction[i] = 1.0;          // only use refValue
            refValue[i] = internalCellValue; // not used
            refGradient[i] = zero<ValueType>();
        },
        "extrapolateValue"
    );
}
}

template<typename ValueType>
class Extrapolated :
    public VolumeBoundaryFactory<ValueType>::template Register<Extrapolated<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<Extrapolated<ValueType>>;

public:

    using ExtrapolatedType = Extrapolated<ValueType>;

    Extrapolated(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition([[maybe_unused]] Field<ValueType>& domainVector) final
    {
        detail::extrapolateValue(domainVector, mesh_, this->range());
    }

    static std::string name() { return "extrapolated"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Extrapolated>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};
}
