// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"

#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
DdtOperator<ValueType>::~DdtOperator()
{}

template<typename ValueType>
DdtOperator<ValueType>::DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff(1.0), field, termType),
      sparsityPattern_(la::SparsityPattern::readOrCreate(field.mesh())), scheme_(scheme) {};

template<typename ValueType>
void DdtOperator<ValueType>::explicitOperation(Vector<ValueType>& source, scalar, scalar dt) const
{
    const scalar dtInver = 1.0 / dt;
    const auto vol = this->getVector().mesh().cellVolumes().view();
    auto [sourceView, field, oldVector] =
        views(source, this->field_.internalVector(), oldTime(this->field_).internalVector());

    parallelFor(
        source.exec(),
        source.range(),
        KOKKOS_LAMBDA(const localIdx celli) {
            sourceView[celli] += dtInver * (field[celli] - oldVector[celli]) * vol[celli];
        },
        "ddtOpertator::explicitOperation"
    );
}

template<typename ValueType>
void DdtOperator<ValueType>::implicitOperation(
    la::LinearSystem<ValueType, localIdx>& ls, scalar, scalar dt
) const
{
    const auto vol = this->getVector().mesh().cellVolumes().view();
    const auto operatorScaling = this->getCoefficient();
    const auto [diagOffs, oldVector] =
        views(getSparsityPattern().diagOffset(), oldTime(this->field_).internalVector());
    auto [matrix, rhs] = ls.view();

    const scalar a0 = scheme_.a0(dt);
    const scalar a1 = scheme_.a1(dt);

    parallelFor(
        ls.exec(),
        {0, oldVector.size()},
        KOKKOS_LAMBDA(const localIdx celli) {
            const auto idx = matrix.rowOffs[celli] + diagOffs[celli];
            const auto commonCoef = operatorScaling[celli] * vol[celli];
            matrix.values[idx] += commonCoef * a0 * one<ValueType>();
            rhs[celli] += commonCoef * a1 * oldVector[celli];
        },
        "ddtOpertator::implicitOperation"
    );
}

// instantiate the template class
template class DdtOperator<scalar>;
template class DdtOperator<Vec3>;

};
