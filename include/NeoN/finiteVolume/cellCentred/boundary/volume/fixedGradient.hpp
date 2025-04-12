// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

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
    DomainField<ValueType>& domainField,
    const UnstructuredMesh& mesh,
    std::pair<size_t, size_t> range,
    ValueType fixedGradient
)
{
    const auto iField = domainField.internalField().view();

    auto [refGradient, value, valueFraction, refValue, faceCells, deltaCoeffs] = spans(
        domainField.boundaryField().refGrad(),
        domainField.boundaryField().value(),
        domainField.boundaryField().valueFraction(),
        domainField.boundaryField().refValue(),
        mesh.boundaryMesh().faceCells(),
        mesh.boundaryMesh().deltaCoeffs()
    );


    NeoN::parallelFor(
        domainField.exec(),
        range,
        KOKKOS_LAMBDA(const size_t i) {
            refGradient[i] = fixedGradient;
            // operator / is not defined for all ValueTypes
            value[i] =
                iField[static_cast<size_t>(faceCells[i])] + fixedGradient * (1 / deltaCoeffs[i]);
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

    FixedGradient(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID), mesh_(mesh),
          fixedGradient_(dict.get<ValueType>("fixedGradient"))
    {}

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) final
    {
        detail::setGradientValue(domainField, mesh_, this->range(), fixedGradient_);
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
