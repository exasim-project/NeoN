// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors


#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
DdtOperator<ValueType>::DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff(1.0), field, termType),
      sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {};

template<typename ValueType>
void DdtOperator<ValueType>::explicitOperation(Field<ValueType>& source, scalar, scalar dt) const
{
    const scalar dtInver = 1.0 / dt;
    const auto vol = this->getField().mesh().cellVolumes().view();
    auto [sourceSpan, field, oldField] =
        spans(source, this->field_.internalField(), oldTime(this->field_).internalField());

    NeoN::parallelFor(
        source.exec(),
        source.range(),
        KOKKOS_LAMBDA(const size_t celli) {
            sourceSpan[celli] += dtInver * (field[celli] - oldField[celli]) * vol[celli];
        }
    );
}

template<typename ValueType>
void DdtOperator<ValueType>::implicitOperation(
    la::LinearSystem<ValueType, localIdx>& ls, scalar, scalar dt
)
{
    const scalar dtInver = 1.0 / dt;
    const auto vol = this->getField().mesh().cellVolumes().view();
    const auto operatorScaling = this->getCoefficient();
    const auto [diagOffs, oldField] =
        spans(sparsityPattern_->diagOffset(), oldTime(this->field_).internalField());
    auto [matrix, rhs] = ls.view();

    NeoN::parallelFor(
        ls.exec(),
        {0, oldField.size()},
        KOKKOS_LAMBDA(const size_t celli) {
            std::size_t idx = matrix.rowOffs[celli] + diagOffs[celli];
            const auto commonCoef = operatorScaling[celli] * vol[celli] * dtInver;
            matrix.values[idx] += commonCoef * one<ValueType>();
            rhs[celli] += commonCoef * oldField[celli];
        }
    );
}

// instantiate the template class
template class DdtOperator<scalar>;
template class DdtOperator<Vector>;

};
