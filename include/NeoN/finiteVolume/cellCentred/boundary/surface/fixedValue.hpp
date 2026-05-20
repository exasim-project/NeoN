// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary
{
namespace detail
{
// Without this function the compiler warns that calling a __host__ function
// from a __device__ function is not allowed
template<typename ValueType>
void setFixedValue(
    Field<ValueType>& domainVector, std::pair<size_t, size_t> range, ValueType fixedValue
)
{
    auto refValue = domainVector.boundaryData().refValue().view();
    auto value = domainVector.boundaryData().value().view();

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            refValue[i] = fixedValue;
            value[i] = fixedValue;
        },
        "setFixedValueSurface"
    );
}
}

template<typename ValueType>
class FixedValue :
    public SurfaceBoundaryFactory<ValueType>::template Register<FixedValue<ValueType>>
{
    using Base = SurfaceBoundaryFactory<ValueType>::template Register<FixedValue<ValueType>>;

public:

    FixedValue(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID), fixedValue_(dict.get<ValueType>("fixedValue"))
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) override
    {
        detail::setFixedValue(domainVector, this->range(), fixedValue_);
    }

    static std::string name() { return "fixedValue"; }

    static std::string doc() { return "Set a fixed value on the boundary"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<SurfaceBoundaryFactory<ValueType>> clone() const override
    {
        return std::make_unique<FixedValue>(*this);
    }

private:

    ValueType fixedValue_;
};

}
