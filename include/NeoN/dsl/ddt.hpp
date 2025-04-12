// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/dsl/spatialOperator.hpp"

namespace NeoN::dsl::temporal
{

template<typename FieldType>
class Ddt : public OperatorMixin<FieldType>
{

public:

    Ddt(FieldType& field) : OperatorMixin<FieldType>(field.exec(), field, Operator::Type::Implicit)
    {}

    std::string getName() const { return "TimeOperator"; }

    void explicitOperation(
        [[maybe_unused]] Field<scalar>& source,
        [[maybe_unused]] scalar t,
        [[maybe_unused]] scalar dt
    )
    {
        NF_ERROR_EXIT("Not implemented");
    }

    void implicitOperation(
        [[maybe_unused]] la::LinearSystem<scalar, localIdx>& ls,
        [[maybe_unused]] scalar t,
        [[maybe_unused]] scalar dt
    )
    {
        NF_ERROR_EXIT("Not implemented");
    }
};

/* @brief factory function to create a Ddt term as ddt() */
template<typename FieldType>
Ddt<FieldType> ddt(FieldType& in)
{
    return Ddt(in);
};

} // namespace NeoN
