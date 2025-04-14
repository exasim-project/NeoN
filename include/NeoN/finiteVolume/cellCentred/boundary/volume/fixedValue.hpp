// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{
// TODO move to source
// Without this function the compiler warns that calling a __host__ function
// from a __device__ function is not allowed
template<typename ValueType>
void setFixedValue(
    Field<ValueType>& domainVector, std::pair<size_t, size_t> range, ValueType fixedValue
)
{
    auto [refGradient, value, valueFraction, refValue] = spans(
        domainVector.boundaryVector().refGrad(),
        domainVector.boundaryVector().value(),
        domainVector.boundaryVector().valueFraction(),
        domainVector.boundaryVector().refValue()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        KOKKOS_LAMBDA(const localIdx i) {
            refValue[i] = fixedValue;
            value[i] = fixedValue;
            valueFraction[i] = 1.0;      // only used refValue
            refGradient[i] = fixedValue; // not used
        }
    );
}

}

template<typename ValueType>
class FixedValue : public VolumeBoundaryFactory<ValueType>::template Register<FixedValue<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<FixedValue<ValueType>>;

public:

    FixedValue(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID), fixedValue_(dict.get<ValueType>("fixedValue"))
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::setFixedValue(domainVector, this->range(), fixedValue_);
    }

    static std::string name() { return "fixedValue"; }

    static std::string doc() { return "Set a fixed value on the boundary"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<FixedValue>(*this);
    }

private:

    ValueType fixedValue_;
};

}
