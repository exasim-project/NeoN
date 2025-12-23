// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"
#include "NeoN/dsl/temporalOperator.hpp"

namespace NeoN::dsl
{

template<typename FieldType>
TemporalOperator<typename FieldType::VectorValueType> ddt(FieldType& field)
{
    using ValueType = typename FieldType::VectorValueType;

    auto fvOp = NeoN::finiteVolume::cellCentred::DdtOperator<ValueType>(
        NeoN::dsl::Operator::Type::Implicit, field
    );

    return TemporalOperator<ValueType>(std::move(fvOp));
}

} // namespace NeoN
