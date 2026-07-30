// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

#include "NeoN/blockAmr/linearAlgebra/coefficients.hpp"

namespace blockamr::la
{

/* @class Operator
 * @brief Value-semantic holder for any IsOperator type; erasure shape as la::Matrix. Its
 *        one difference is the point: `assemble` is PRIVATE, friending only LinearSystem,
 *        so `system += op` is the only way in (asserted in coefficientsConcepts.cpp).
 */
class Operator
{
public:

    // Same guard as Matrix's ctor, redundant today because `assemble`'s privacy keeps
    // Operator out of IsOperator; kept so making it public errors instead of nesting.
    template<IsOperator T>
        requires(!std::same_as<std::remove_cvref_t<T>, Operator>)
    Operator(T op) : model_(std::make_unique<Model<T>>(std::move(op)))
    {}

    Operator(const Operator& op) : model_(op.model_->clone()) {}

    Operator(Operator&& op) = default;

    Operator& operator=(const Operator& op)
    {
        model_ = op.model_->clone();
        return *this;
    }

    Operator& operator=(Operator&& op) = default;

    ~Operator() = default;

private:

    // The gate. `system += op` is the only way in.
    friend class LinearSystem;

    void assemble(Coefficients c) const { model_->assemble(c); }

    struct Concept
    {
        virtual ~Concept() = default;

        virtual void assemble(Coefficients c) const = 0;

        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    template<IsOperator T>
    struct Model : Concept
    {
        Model(T cls) : cls_(std::move(cls)) {}

        void assemble(Coefficients c) const override { cls_.assemble(c); }

        std::unique_ptr<Concept> clone() const override { return std::make_unique<Model<T>>(cls_); }

        T cls_;
    };

    std::unique_ptr<Concept> model_;
};

} // namespace blockamr::la
