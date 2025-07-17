// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceTerm.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
SourceTerm<ValueType>::SourceTerm(
    dsl::Operator::Type termType, VolumeField<scalar>& coefficients, VolumeField<ValueType>& field
)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff {1.0}, field, termType),
      coefficients_(coefficients),
      sparsityPattern_(la::SparsityPattern::readOrCreate(field.mesh())) {};

template<typename ValueType>
void SourceTerm<ValueType>::explicitOperation(Vector<ValueType>& source) const
{
    auto operatorScaling = this->getCoefficient();
    auto [sourceView, fieldView, coeff] =
        views(source, this->field_.internalVector(), coefficients_.internalVector());
    NeoN::parallelFor(
        source.exec(),
        source.range(),
        KOKKOS_LAMBDA(const localIdx celli) {
            sourceView[celli] += operatorScaling[celli] * coeff[celli] * fieldView[celli];
        }
    );
}

template<typename ValueType>
void SourceTerm<ValueType>::implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) const
{
    const auto operatorScaling = this->getCoefficient();
    const auto vol = coefficients_.mesh().cellVolumes().view();
    const auto [diagOffs, coeff] =
        views(getSparsityPattern().diagOffset(), coefficients_.internalVector());
    auto [matrix, rhs] = ls.view();

    NeoN::parallelFor(
        ls.exec(),
        {0, coeff.size()},
        KOKKOS_LAMBDA(const localIdx celli) {
            localIdx idx = matrix.rowOffs[celli] + diagOffs[celli];
            matrix.values[idx] +=
                operatorScaling[celli] * coeff[celli] * vol[celli] * one<ValueType>();
        }
    );
}


// instantiate the template class
template class SourceTerm<scalar>;
template class SourceTerm<Vec3>;
};
