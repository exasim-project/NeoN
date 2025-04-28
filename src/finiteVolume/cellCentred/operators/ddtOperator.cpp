// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors


#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"

#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
DdtOperator<ValueType>::DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff(1.0), field, termType),
      sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {};

template<typename ValueType>
void DdtOperator<ValueType>::explicitOperation(Vector<ValueType>& source, scalar, scalar dt) const
{
    const scalar dtInver = 1.0 / dt;
    const auto vol = this->getVector().mesh().cellVolumes().view();
    auto [sourceView, field, oldVector] =
        views(source, this->field_.internalVector(), oldTime(this->field_).internalVector());

    NeoN::parallelFor(
        source.exec(),
        source.range(),
        KOKKOS_LAMBDA(const localIdx celli) {
            sourceView[celli] += dtInver * (field[celli] - oldVector[celli]) * vol[celli];
        }
    );
}

template<typename ValueType>
void DdtOperator<ValueType>::implicitOperation(la::LinearSystem<ValueType>& ls, scalar, scalar dt)
{
    const scalar dtInver = 1.0 / dt;
    const auto vol = this->getVector().mesh().cellVolumes().view();
    const auto operatorScaling = this->getCoefficient();
    const auto [diagOffs, oldVector] =
        views(sparsityPattern_->diagOffset(), oldTime(this->field_).internalVector());
    auto [matrix, rhs] = ls.view();

    NeoN::parallelFor(
        ls.exec(),
        {0, oldVector.size()},
        KOKKOS_LAMBDA(const localIdx celli) {
            const auto idx = matrix.rowOffs[celli] + diagOffs[celli];
            const auto commonCoef = operatorScaling[celli] * vol[celli] * dtInver;
            matrix.values[idx] += commonCoef * one<ValueType>();
            rhs[celli] += commonCoef * oldVector[celli];
        }
    );
}

// instantiate the template class
template class DdtOperator<scalar>;
template class DdtOperator<Vec3>;

};
