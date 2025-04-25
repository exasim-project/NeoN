// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/dsl/temporalOperator.hpp"

namespace NeoN::dsl::temporal
{

template<typename VectorType>
class Ddt : public OperatorMixin<VectorType>
{

public:

    using VectorValueType = scalar;

    Ddt(VectorType& field, dsl::Operator::Type type)
        : OperatorMixin<VectorType>(field.exec(), {}, field, type)
    {}

    std::string getName() const { return "TimeOperator"; }

    void read(const Input&) {}

    void explicitOperation(Vector<scalar>&, scalar, scalar) const
    {
        NF_ERROR_EXIT("Not implemented");
        // const scalar dtInver = 1.0 / dt;
        // const auto vol = this->getVector().mesh().cellVolumes().view();
        // auto [sourceView, field, oldVector] =
        //     views(source, this->field_.internalVector(), oldTime(this->field_).internalVector());

        // NeoN::parallelFor(
        //     source.exec(),
        //     source.range(),
        //     KOKKOS_LAMBDA(const localIdx celli) {
        //         sourceView[celli] += dtInver * (field[celli] - oldVector[celli]) * vol[celli];
        //     }
        // );
    }

    void implicitOperation(la::LinearSystem<scalar, localIdx>&, scalar, scalar)
    {
        NF_ERROR_EXIT("Not implemented");
        // const scalar dtInver = 1.0 / dt;
        // const auto vol = this->getVector().mesh().cellVolumes().view();
        // const auto operatorScaling = this->getCoefficient();
        // const auto [diagOffs, oldVector] =
        //     views(sparsityPattern_->diagOffset(), oldTime(this->field_).internalVector());
        // auto [matrix, rhs] = ls.view();

        // NeoN::parallelFor(
        //     ls.exec(),
        //     {0, oldVector.size()},
        //     KOKKOS_LAMBDA(const localIdx celli) {
        //         const auto idx = matrix.rowOffs[celli] + diagOffs[celli];
        //         const auto commonCoef = operatorScaling[celli] * vol[celli] * dtInver;
        //         matrix.values[idx] += commonCoef * one<ValueType>();
        //         rhs[celli] += commonCoef * oldVector[celli];
        //     }
        // );
    }
};

} // namespace NeoN

namespace NeoN::dsl::imp
{
/* @brief factory function to create a Ddt term as ddt() */
template<typename VectorType>
TemporalOperator<scalar> ddt(VectorType& in)
{
    return temporal::Ddt(in, Operator::Type::Implicit);
};

}

namespace NeoN::dsl::exp
{
/* @brief factory function to create a Ddt term as ddt() */
template<typename VectorType>
TemporalOperator<scalar> ddt(VectorType& in)
{
    return temporal::Ddt(in, Operator::Type::Explicit);
};

}
