// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#pragma once

#include "NeoN/dsl/coeff.hpp"

namespace NeoN::dsl
{

class Operator
{
public:

    enum class Type
    {
        Implicit,
        Explicit
    };
};


/* @class OperatorMixin
 * @brief A mixin class to simplify implementations of concrete operators
 * in NeoNs dsl
 *
 * @ingroup dsl
 */
template<typename FieldType>
class OperatorMixin
{

public:

    OperatorMixin(const Executor exec, const Coeff& coeffs, FieldType& field, Operator::Type type)
        : exec_(exec), coeffs_(coeffs), field_(field), type_(type) {};


    Operator::Type getType() const { return type_; }

    virtual ~OperatorMixin() = default;

    virtual const Executor& exec() const final { return exec_; }

    Coeff& getCoefficient() { return coeffs_; }

    const Coeff& getCoefficient() const { return coeffs_; }

    FieldType& getField() { return field_; }

    const FieldType& getField() const { return field_; }

    /* @brief Given an input this function reads required coeffs */
    void build([[maybe_unused]] const Input& input) {}

protected:

    const Executor exec_; //!< Executor associated with the field. (CPU, GPU, openMP, etc.)

    Coeff coeffs_;

    FieldType& field_;

    Operator::Type type_;
};

} // namespace NeoN::dsl
