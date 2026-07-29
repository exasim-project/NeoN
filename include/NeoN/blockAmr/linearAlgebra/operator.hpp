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
 * The second erasure in this component, and deliberately the SAME shape as
 * la::Matrix (linearAlgebra/matrix.hpp) and NeoN's dsl::SpatialOperator: a
 * private abstract `Concept` naming exactly the surface, a `Model<T>` holding
 * one T BY VALUE, and copy through `clone()`. A reader who knows one knows all
 * three.
 *
 * ONE structural difference from Matrix, and it is the point of the class:
 * `assemble` is PRIVATE, with `friend class LinearSystem`. Everything the
 * erasure can do is therefore reachable only from `system += op`. A concrete
 * operator's own `assemble` is public -- IsOperator requires it -- but its
 * argument, a `Coefficients`, has a private constructor whose only friend is
 * also `LinearSystem` (coefficients.hpp), so calling it directly is not
 * something a caller can spell either. The two halves together are the gate:
 * nothing can be assembled outside `+=`, and nothing outside `+=` can even
 * produce the argument.
 *
 * A consequence worth stating because it differs from Matrix: `Operator` does
 * NOT satisfy `IsOperator` (the concept looks for a public `assemble`), so there
 * is no `static_assert(IsOperator<Operator>)` here to check the forwarding
 * surface against the concept the way matrix.hpp does. The negative form is
 * asserted instead, in linearAlgebra/coefficientsConcepts.cpp -- if `assemble`
 * ever became public, that assertion fires and the gate's loss is a compile
 * error rather than a silent widening.
 */
class Operator
{
public:

    // The constrained converting constructor, carried over from Matrix verbatim.
    // There it is load-bearing: Matrix satisfies IsMatrix, so without the
    // `requires` a `Matrix b {a};` on a NON-const lvalue prefers this template
    // over the copy constructor and wraps a Matrix inside a Matrix. Here the
    // privacy of `assemble` already keeps Operator out of IsOperator, so the
    // template is not even a candidate and the guard is currently redundant --
    // it is kept because the day someone makes `assemble` public "just to test
    // it", the guard is the difference between a compile error at that change
    // and a silently nesting Operator.
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
