// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

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
void setGradientValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    ValueType fixedGradient
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
        KOKKOS_LAMBDA(const localIdx i) {
            refGradient[i] = fixedGradient;
            // operator / is not defined for all ValueTypes
            value[i] = iVector[faceCells[i]] + fixedGradient * (1 / deltaCoeffs[i]);
            valueFraction[i] = 0.0;          // only use refGrad
            refValue[i] = zero<ValueType>(); // not used
        },
        "setGradientValue"
    );
}
}

template<typename ValueType>
class FixedGradient :
    public VolumeBoundaryFactory<ValueType>::template Register<FixedGradient<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<FixedGradient<ValueType>>;

public:

    using FixedGradientType = FixedGradient<ValueType>;

    FixedGradient(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true}), mesh_(mesh),
          fixedGradient_(dict.get<ValueType>("fixedGradient"))
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::setGradientValue(domainVector, mesh_, this->range(), fixedGradient_);
    }

    static std::string name() { return "fixedGradient"; }

    static std::string doc() { return "Set a fixed gradient on the boundary."; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<FixedGradient>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    ValueType fixedGradient_;
};

}
