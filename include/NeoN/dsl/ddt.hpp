// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/dsl/spatialOperator.hpp"

namespace NeoN::dsl::temporal
{

template<typename VectorType>
class Ddt : public OperatorMixin<VectorType>
{

public:

    Ddt(VectorType& field)
        : OperatorMixin<VectorType>(field.exec(), field, Operator::Type::Implicit)
    {}

    std::string getName() const { return "TimeOperator"; }

    void explicitOperation(Vector<scalar>&, scalar, scalar) const
    {
        NF_ERROR_EXIT("Not implemented");
    }

    void implicitOperation(la::LinearSystem<scalar, localIdx>&, scalar, scalar) const
    {
        NF_ERROR_EXIT("Not implemented");
    }
};

/* @brief factory function to create a Ddt term as ddt() */
template<typename VectorType>
Ddt<VectorType> ddt(VectorType& in)
{
    return Ddt(in);
};

} // namespace NeoN
