// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceTerm.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
SourceTerm<ValueType>::~SourceTerm()
{}

template<typename ValueType>
SourceTerm<ValueType>::SourceTerm(
    dsl::Operator::Type termType,
    const VolumeField<scalar>& coefficients,
    const VolumeField<ValueType>& field
)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff {1.0}, field, termType),
      coefficients_(coefficients) {};

template<typename ValueType>
void SourceTerm<ValueType>::explicitOperation(Vector<ValueType>& source) const
{
    auto operatorScaling = this->getCoefficient();
    auto [sourceView, fieldView, coeff] =
        views(source, this->field_.internalVector(), coefficients_.internalVector());
    NeoN::parallelFor(
        source.exec(),
        source.range(),
        NEON_LAMBDA(const localIdx celli) {
            sourceView[celli] += operatorScaling[celli] * coeff[celli] * fieldView[celli];
        },
        "sourceTerm::explicitOperation"
    );
}

template<typename ValueType>
void SourceTerm<ValueType>::implicitOperation(la::LinearSystem<ValueType>& ls) const
{
    const auto matIt = ls.faceToMatrixAddress();
    const auto operatorScaling = this->getCoefficient();
    const auto vol = coefficients_.mesh().cellVolumes().view();
    const auto [diagOffs, coeff] = views(matIt->diagOffset(), coefficients_.internalVector());
    auto values = ls.matrix().values().view();
    auto [colIdx, rowOffs] = ls.matrix().sparsity()->view();

    NeoN::parallelFor(
        ls.exec(),
        {0, coeff.size()},
        NEON_LAMBDA(const localIdx celli) {
            localIdx idx = rowOffs[celli] + diagOffs[celli];
            values[idx] += operatorScaling[celli] * coeff[celli] * vol[celli] * one<ValueType>();
        },
        "sourceTerm::implicitOperation"
    );
}


// instantiate the template class
template class SourceTerm<scalar>;
template class SourceTerm<Vec3>;
};
