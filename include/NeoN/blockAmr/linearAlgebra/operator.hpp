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
 * @brief Value-semantic holder for any type satisfying IsOperator.
 *
 * Erasure shape: see la::Matrix (linearAlgebra/matrix.hpp).
 *
 * ONE structural difference from Matrix, and it is the point of the class:
 * `assemble` is PRIVATE with `friend class LinearSystem`, so the erasure is
 * reachable only from `system += op`. The other half of the gate is that a
 * `Coefficients` has a private constructor friending only LinearSystem
 * (coefficients.hpp), so nothing outside `+=` can even produce the argument.
 *
 * Consequently `Operator` does NOT satisfy `IsOperator` (which wants a public
 * `assemble`), so there is no `static_assert(IsOperator<Operator>)` here as in
 * matrix.hpp; the NEGATIVE form is asserted in
 * linearAlgebra/coefficientsConcepts.cpp so that making `assemble` public breaks
 * the build instead of silently widening the gate.
 */
class Operator
{
public:

    // Same guard as Matrix's ctor (matrix.hpp), currently redundant here because
    // the privacy of `assemble` keeps Operator out of IsOperator. Kept so that
    // making `assemble` public gives a compile error rather than a nesting Operator.
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
