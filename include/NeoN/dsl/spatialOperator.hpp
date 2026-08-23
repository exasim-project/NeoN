// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <concepts>

#include "NeoN/core/error.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/coeff.hpp"
#include "NeoN/dsl/operator.hpp"

namespace la = NeoN::la;

namespace NeoN::dsl
{

template<typename T>
concept HasExplicitOperator = requires(T const t) {
    {
        t.explicitOperation(std::declval<Vector<typename T::VectorValueType>&>())
    } -> std::same_as<void>;
};

template<typename T>
concept HasImplicitOperator = requires(T const t) {
    {
        t.implicitOperation(std::declval<la::LinearSystem<typename T::VectorValueType>&>())
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};

/* @brief Concept satisfied when T can assemble into a LinearSystem whose matrix
 *        coefficients are scalar while the RHS holds T's field value type
 *        (segregated vector-solve form). Only meaningful when VectorValueType != scalar.
 */
template<typename T>
concept HasImplicitOperatorScalarMtx = requires(T const t) {
    {
        t.implicitOperation(std::declval<la::LinearSystem<scalar, typename T::VectorValueType>&>())
    } -> std::same_as<void>;
};

/* @brief Concept satisfied when T can assemble into a native ELL-backed LinearSystem
 *        (scalar matrix, scalar rhs). Matches SourceTerm::implicitOperation<SystemMatrixType>'s
 *        existing template shape directly -- SourceTerm needs no changes to satisfy this.
 */
template<typename T>
concept HasImplicitOperatorELL = requires(T const t) {
    {
        t.implicitOperation(
            std::declval<la::LinearSystem<scalar, scalar, la::ELLMatrix<scalar, localIdx>>&>()
        )
    } -> std::same_as<void>;
};

/* @brief Concept satisfied when T can assemble into a native ELL-backed LinearSystem in the
 *        segregated vector-solve form (scalar matrix, T::VectorValueType rhs). Only meaningful
 *        when VectorValueType != scalar -- for scalar T this collides with HasImplicitOperatorELL
 *        (same declval type), which is why the dispatch below only ever calls this branch when
 *        ValueType != scalar (mirrors HasImplicitOperatorScalarMtx's CSR counterpart).
 */
template<typename T>
concept HasImplicitOperatorScalarMtxELL = requires(T const t) {
    {
        t.implicitOperation(std::declval<la::LinearSystem<
                                scalar,
                                typename T::VectorValueType,
                                la::ELLMatrix<scalar, localIdx>>&>())
    } -> std::same_as<void>;
};

template<typename T>
concept IsSpatialOperator =
    HasExplicitOperator<T> || HasImplicitOperator<T> || HasImplicitOperatorScalarMtx<T>
    || HasImplicitOperatorELL<T> || HasImplicitOperatorScalarMtxELL<T>;

/* @class SpatialOperator
 * @brief A class to represent an operator in NeoNs dsl
 *
 * The design here is based on the type erasure design pattern
 * see https://www.youtube.com/watch?v=4eeESJQk-mw
 *
 * Motivation for using type erasure is that concrete implementation
 * of Operators e.g Divergence, Laplacian, etc can be stored in a vector of
 * Operators
 *
 * @ingroup dsl
 */
template<typename ValueType>
class SpatialOperator
{
public:

    using VectorValueType = ValueType;

    template<IsSpatialOperator T>
    SpatialOperator(T cls) : model_(std::make_unique<OperatorModel<T>>(std::move(cls)))
    {}

    SpatialOperator(const SpatialOperator& eqnOperator) : model_(eqnOperator.model_->clone()) {}

    SpatialOperator(SpatialOperator&& eqnOperator) : model_(std::move(eqnOperator.model_)) {}

    SpatialOperator& operator=(const SpatialOperator& eqnOperator)
    {
        model_ = eqnOperator.model_->clone();
        return *this;
    }

    void explicitOperation(Vector<ValueType>& source) const { model_->explicitOperation(source); }

    void implicitOperation(la::LinearSystem<ValueType>& ls) const { model_->implicitOperation(ls); }

    /* @brief Implicit assembly into a scalar-matrix / ValueType-rhs linear system
     *        (segregated vector-solve form). Disabled when ValueType == scalar to
     *        avoid colliding with the same-type overload above.
     */
    void implicitOperation(la::LinearSystem<scalar, ValueType>& ls) const
        requires(!std::is_same_v<ValueType, scalar>)
    {
        model_->implicitOperationScalarMtx(ls);
    }

    /* @brief Implicit assembly into a native ELL-backed (scalar matrix, scalar rhs) linear
     *        system. A fixed, concrete parameter type distinct from both overloads above
     *        (different SystemMatrixType), so this coexists without ambiguity regardless of
     *        ValueType -- only operators that actually support ELL (see HasImplicitOperatorELL)
     *        do anything here; others throw via implicitOperationELL's default below.
     */
    void implicitOperation(la::LinearSystem<scalar, scalar, la::ELLMatrix<scalar, localIdx>>& ls
    ) const
    {
        model_->implicitOperationELL(ls);
    }

    /* @brief Implicit assembly into a native ELL-backed LinearSystem in the segregated
     *        vector-solve form (scalar matrix, ValueType rhs). Disabled when ValueType == scalar
     *        to avoid colliding with the same-type ELL overload above (identical declval type).
     */
    void implicitOperation(la::LinearSystem<scalar, ValueType, la::ELLMatrix<scalar, localIdx>>& ls
    ) const
        requires(!std::is_same_v<ValueType, scalar>)
    {
        model_->implicitOperationScalarMtxELL(ls);
    }

    /* returns the fundamental type of an operator, ie explicit, implicit */
    Operator::Type getType() const { return model_->getType(); }

    std::string getName() const { return model_->getName(); }

    Coeff& getCoefficient() { return model_->getCoefficient(); }

    Coeff getCoefficient() const { return model_->getCoefficient(); }

    Dictionary getConfig() const { return model_->getConfig(); }

    /* @brief Given an input this function reads required properties */
    void read(const Input& input) { model_->read(input); }

    /* @brief Get the executor */
    const Executor& exec() const { return model_->exec(); }


private:

    /* @brief Base class defining the concept of a term. This effectively
     * defines what functions need to be implemented by a concrete Operator implementation
     * */
    struct OperatorConcept
    {
        virtual ~OperatorConcept() = default;

        virtual void explicitOperation(Vector<ValueType>& source) const = 0;

        virtual void implicitOperation(la::LinearSystem<ValueType>& ls) const = 0;

        /* @brief Implicit assembly into LinearSystem<scalar, ValueType> for the
         *        scalar-matrix / ValueType-rhs (segregated vector-solve) form.
         *        Concrete operators that don't support this form leave it as a no-op.
         */
        virtual void implicitOperationScalarMtx(la::LinearSystem<scalar, ValueType>& ls) const = 0;

        /* @brief Implicit assembly into a native ELL-backed (scalar matrix, scalar rhs)
         *        linear system. Concrete operators that don't support ELL yet throw
         *        (see OperatorModel's implementation), same as implicitOperationScalarMtx.
         */
        virtual void
        implicitOperationELL(la::LinearSystem<scalar, scalar, la::ELLMatrix<scalar, localIdx>>& ls
        ) const = 0;

        /* @brief Implicit assembly into a native ELL-backed LinearSystem, segregated
         *        vector-solve form (scalar matrix, ValueType rhs). Concrete operators that
         *        don't support this yet throw, same as implicitOperationELL /
         *        implicitOperationScalarMtx.
         */
        virtual void implicitOperationScalarMtxELL(
            la::LinearSystem<scalar, ValueType, la::ELLMatrix<scalar, localIdx>>& ls
        ) const = 0;

        /* @brief Given an input this function reads required coeffs */
        virtual void read(const Input& input) = 0;

        /* returns the name of the operator */
        virtual std::string getName() const = 0;

        /* returns the fundamental type of an operator, ie explicit, implicit */
        virtual Operator::Type getType() const = 0;

        /* @brief get the associated coefficient for this term */
        virtual Coeff& getCoefficient() = 0;

        /* @brief get the associated coefficient for this term */
        virtual Coeff getCoefficient() const = 0;

        /* @brief Get the config of operator*/
        virtual Dictionary getConfig() const = 0;

        /* @brief Get the executor */
        virtual const Executor& exec() const = 0;

        // The Prototype Design Pattern
        virtual std::unique_ptr<OperatorConcept> clone() const = 0;
    };

    // Templated derived class to implement the type-specific behavior
    template<typename ConcreteOperatorType>
    struct OperatorModel : OperatorConcept
    {
        /* @brief build with concrete operator */
        OperatorModel(ConcreteOperatorType concreteOp) : concreteOp_(std::move(concreteOp)) {}

        /* returns the name of the operator */
        std::string getName() const override { return concreteOp_.getName(); }

        virtual void explicitOperation(Vector<ValueType>& source) const override
        {
            if constexpr (HasExplicitOperator<ConcreteOperatorType>)
            {
                concreteOp_.explicitOperation(source);
            }
        }

        virtual void implicitOperation(la::LinearSystem<ValueType>& ls) const override
        {
            if constexpr (HasImplicitOperator<ConcreteOperatorType>)
            {
                concreteOp_.implicitOperation(ls);
            }
        }

        virtual void implicitOperationScalarMtx(la::LinearSystem<scalar, ValueType>& ls
        ) const override
        {
            if constexpr (HasImplicitOperatorScalarMtx<ConcreteOperatorType>)
            {
                concreteOp_.implicitOperation(ls);
            }
            else
            {
                // Reached only for an implicit operator that lacks the scalar-matrix
                // (segregated vector-solve) overload. Silently skipping it would drop its
                // contribution and yield a wrong system, so fail fast instead.
                NF_ERROR_EXIT(
                    "Operator '" << getName()
                                 << "' does not support scalar-matrix (segregated) assembly."
                );
            }
        }

        virtual void
        implicitOperationELL(la::LinearSystem<scalar, scalar, la::ELLMatrix<scalar, localIdx>>& ls
        ) const override
        {
            if constexpr (HasImplicitOperatorELL<ConcreteOperatorType>)
            {
                concreteOp_.implicitOperation(ls);
            }
            else
            {
                NF_ERROR_EXIT("Operator '" << getName() << "' does not support ELL assembly.");
            }
        }

        virtual void implicitOperationScalarMtxELL(
            la::LinearSystem<scalar, ValueType, la::ELLMatrix<scalar, localIdx>>& ls
        ) const override
        {
            if constexpr (HasImplicitOperatorScalarMtxELL<ConcreteOperatorType>)
            {
                concreteOp_.implicitOperation(ls);
            }
            else
            {
                NF_ERROR_EXIT(
                    "Operator '" << getName()
                                 << "' does not support scalar-matrix (segregated) ELL assembly."
                );
            }
        }

        /* @brief Given an input this function reads required coeffs */
        virtual void read(const Input& input) override { concreteOp_.read(input); }

        /* returns the fundamental type of an operator, ie explicit, implicit, temporal */
        Operator::Type getType() const override { return concreteOp_.getType(); }

        /* @brief Get the executor */
        const Executor& exec() const override { return concreteOp_.exec(); }

        /* @brief get the associated coefficient for this term */
        virtual Coeff& getCoefficient() override { return concreteOp_.getCoefficient(); }

        /* @brief get the associated coefficient for this term */
        virtual Coeff getCoefficient() const override { return concreteOp_.getCoefficient(); }

        virtual Dictionary getConfig() const override { return concreteOp_.getConfig(); }

        // The Prototype Design Pattern
        std::unique_ptr<OperatorConcept> clone() const override
        {
            return std::make_unique<OperatorModel>(*this);
        }

        ConcreteOperatorType concreteOp_;
    };

    std::unique_ptr<OperatorConcept> model_;
};


template<typename ValueType>
SpatialOperator<ValueType> operator*(scalar scalarCoeff, SpatialOperator<ValueType> rhs)
{
    SpatialOperator<ValueType> result = rhs;
    result.getCoefficient() *= scalarCoeff;
    return result;
}

template<typename ValueType>
SpatialOperator<ValueType>
operator*(const Vector<scalar>& coeffVector, SpatialOperator<ValueType> rhs)
{
    SpatialOperator<ValueType> result = rhs;
    result.getCoefficient() *= Coeff {coeffVector};
    return result;
}

template<typename ValueType>
SpatialOperator<ValueType> operator*(const Coeff& coeff, SpatialOperator<ValueType> rhs)
{
    SpatialOperator<ValueType> result = rhs;
    result.getCoefficient() *= coeff;
    return result;
}

// template<typename CoeffFunction>
//     requires std::invocable<CoeffFunction&, size_t>
// SpatialOperator operator*([[maybe_unused]] CoeffFunction coeffFunc, const SpatialOperator& lhs)
// {
//     // TODO implement
//     NF_ERROR_EXIT("Not implemented");
//     SpatialOperator result = lhs;
//     // if (!result.getCoefficient().useView)
//     // {
//     //     result.setVector(std::make_shared<Vector<scalar>>(result.exec(),
//     result.nCells(), 1.0));
//     // }
//     // map(result.exec(), result.getCoefficient().values, scaleFunc);
//     return result;
// }

} // namespace dsl
