// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceUTerm.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
SourceUTerm<ValueType>::~SourceUTerm()
{}

template<typename ValueType>
SourceUTerm<ValueType>::SourceUTerm(
    dsl::Operator::Type termType, VolumeField<ValueType>& coefficients
)
    : dsl::OperatorMixin<VolumeField<ValueType>>(
        coefficients.exec(), dsl::Coeff {1.0}, coefficients, termType
    ),
      coefficients_(coefficients) {};

template<typename ValueType>
void SourceUTerm<ValueType>::explicitOperation(Vector<ValueType>& source) const
{
    auto operatorScaling = this->getCoefficient();
    auto [sourceView, coeff] = views(source, coefficients_.internalVector());
    NeoN::parallelFor(
        source.exec(),
        source.range(),
        NEON_LAMBDA(const localIdx celli) {
            sourceView[celli] += operatorScaling[celli] * coeff[celli];
        },
        "sourceTerm::explicitOperation"
    );
}

template<typename ValueType>
void SourceUTerm<ValueType>::implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) const
{
    NF_ERROR_EXIT("Not implemented");
}


// instantiate the template class
template class SourceUTerm<scalar>;
template class SourceUTerm<Vec3>;
};
