// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

#include "NeoN/core/error.hpp"
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
    auto [refGradient, value, valueFraction, refValue] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            refValue[i] = fixedValue;
            value[i] = fixedValue;
            valueFraction[i] = 1.0;      // only used refValue
            refGradient[i] = fixedValue; // not used
        },
        "setFixedValueVolume"
    );
}

// Per-face variant: the patch carries one value per boundary face, e.g. an inflow
// profile read from a nonuniform OpenFOAM patch field.
template<typename ValueType>
void setFixedValues(
    Field<ValueType>& domainVector,
    std::pair<size_t, size_t> range,
    const Vector<ValueType>& fixedValues
)
{
    auto [refGradient, value, valueFraction, refValue] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue()
    );
    auto fixedValuesView = fixedValues.view();
    const auto start = static_cast<localIdx>(range.first);

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            auto patchValue = fixedValuesView[i - start];
            refValue[i] = patchValue;
            value[i] = patchValue;
            valueFraction[i] = 1.0;      // only used refValue
            refGradient[i] = patchValue; // not used
        },
        "setFixedValuesVolume"
    );
}

}

template<typename ValueType>
class FixedValue : public VolumeBoundaryFactory<ValueType>::template Register<FixedValue<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<FixedValue<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    FixedValue(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true}),
          fixedValue_(
              dict.contains("fixedValue") ? dict.get<ValueType>("fixedValue") : ValueType {}
          ),
          fixedValues_(
              mesh.exec(),
              dict.contains("fixedValues") ? dict.get<std::vector<ValueType>>("fixedValues")
                                           : std::vector<ValueType> {}
          )
    {
        const bool hasUniform = dict.contains("fixedValue");
        const bool hasPerFace = dict.contains("fixedValues");
        if (!hasUniform && !hasPerFace)
        {
            NF_THROW(
                "fixedValue boundary condition on patch " + std::to_string(patchID)
                + " requires either a uniform 'fixedValue' or a per-face 'fixedValues' entry"
            );
        }
        // Validate on the key being present, not on the list being non-empty: an empty
        // 'fixedValues' on a non-empty patch is malformed per-face data, and testing the
        // size alone would let it through to the uniform fallback and silently apply zero.
        if (hasPerFace && fixedValues_.size() != this->patchSize())
        {
            NF_THROW(
                "'fixedValues' holds " + std::to_string(fixedValues_.size()) + " values but patch "
                + std::to_string(patchID) + " has " + std::to_string(this->patchSize()) + " faces"
            );
        }
    }

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        if (fixedValues_.size() > 0)
        {
            detail::setFixedValues(domainVector, this->range(), fixedValues_);
        }
        else
        {
            detail::setFixedValue(domainVector, this->range(), fixedValue_);
        }
    }

    static std::string name() { return "fixedValue"; }

    std::string getName() const override { return name(); }

    static std::string doc() { return "Set a fixed value on the boundary"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<FixedValue>(*this);
    }

private:

    ValueType fixedValue_;
    Vector<ValueType> fixedValues_; ///< empty unless the patch value is nonuniform
};

}
