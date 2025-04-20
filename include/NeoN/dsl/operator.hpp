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

template<typename TermType, typename ValueType>
concept TermConcept = requires(TermType term, localIdx idx, Vector<ValueType>& target) {
    {
        term.evaluateFace(idx, target)
    } -> std::same_as<void>;
};

template<typename DerivedTerm, typename Type>
class Term
{
public:

    void evaluateFace(localIdx faceIdx, Vector<Type>& target)
    {
        static_cast<DerivedTerm*>(this)->evaluateFace(faceIdx, target);
    }

private:
};

template<typename Type>
class DivOP : public Term<DivOP<Type>, Type>
{
public:

    explicit DivOp() {}

    void evaluateFace(localIdx faceIdx, Vector<Type>& target) {}

private:
};

template<typename Type>
class LapOP : public Term<LapOP<Type>, Type>
{
public:

    explicit LapOP() {}

    void evaluateFace(localIdx faceIdx, Vector<Type>& target) {}

private:
};

template<typename Type, template<typename> class LHSType, template<typename> typename RHSType>
class AddTerm : public Term<AddTerm<Type, LHSType, RHSType>, Type>
{
public:

    explicit AddTerm(const LHSType<Type>& lhs, const LHSType<Type>& rhs) : lhs_(lhs), rhs_(rhs) {}

    void evaluateFace(localIdx faceIdx, Vector<Type>& target) const
    {
        lhs_.evaluate(faceIdx, target);
        rhs_.evaluate(faceIdx, target);
    }

private:

    const LHSType<Type>& lhs_;
    const RHSType<Type>& rhs_;
};

template<typename Type, template<typename> class LHS, template<typename> class RHS>
    requires TermConcept<LHS<Type>, Type> && TermConcept<RHS<Type>, Type>
auto operator+(const LHS<Type>& lhs, const RHS<Type>& rhs)
{
    return AddTerm<Type, LHS, RHS>(lhs, rhs);
}

inline void fictitiousExample()
{
    auto expr = DivOP<scalar>() + LapOP<scalar>() + DivOP<scalar>();
    Vector<scalar> target(CPUExecutor(), 1, 0);
    expr.evaluateFace(1, target);
}

template<typename ValueType>
struct OperationDescriptor
{
    using DeviceExecuteFunc = void (*)(
        const void& params, const SurfaceField<ValueType>& fields, const localIdx faceIdx
    ) KOKKOS_FUNCTION;

    deviceExecuteFunc executeFunc;
    parameterType params;

    KOKKOS_FUNCTION
    void execute(const SurfaceField<ValueType>& fields, const localIdx faceIdx) const
    {
        return executeFunc(params, fields, faceIdx);
    }
};


/* @class OperatorMixin
 * @brief A mixin class to simplify implementations of concrete operators
 * in NeoNs dsl
 *
 * @ingroup dsl
 */
template<typename VectorType>
class OperatorMixin
{

public:

    OperatorMixin(const Executor exec, const Coeff& coeffs, VectorType& field, Operator::Type type)
        : exec_(exec), coeffs_(coeffs), field_(field), type_(type) {};


    Operator::Type getType() const { return type_; }

    virtual ~OperatorMixin() = default;

    virtual const Executor& exec() const final { return exec_; }

    Coeff& getCoefficient() { return coeffs_; }

    const Coeff& getCoefficient() const { return coeffs_; }

    VectorType& getVector() { return field_; }

    const VectorType& getVector() const { return field_; }

    /* @brief Given an input this function reads required coeffs */
    void build([[maybe_unused]] const Input& input) {}

protected:

    const Executor exec_; //!< Executor associated with the field. (CPU, GPU, openMP, etc.)

    Coeff coeffs_;

    VectorType& field_;

    Operator::Type type_;
};

} // namespace NeoN::dsl
