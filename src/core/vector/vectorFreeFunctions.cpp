// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include <Kokkos_Core.hpp>

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/view.hpp"
#include "NeoN/helpers/exceptions.hpp"

namespace NeoN
{

template<typename ValueType>
void scalarMul(Vector<ValueType>& vect, const ValueType value)
    requires requires(ValueType a, ValueType b) { a* b; }
{
    auto viewA = vect.view();
    parallelFor(
        vect, KOKKOS_LAMBDA(const localIdx i) { return viewA[i] * value; }
    );
}

namespace detail
{

template<typename ValueType, typename BinaryOp>
void fieldBinaryOp(
    Vector<ValueType>& a, const Vector<std::type_identity_t<ValueType>>& b, BinaryOp op
)
{
    NeoN_ASSERT_EQUAL_LENGTH(a, b);
    auto viewA = a.view();
    auto viewB = b.view();
    parallelFor(
        a, KOKKOS_LAMBDA(const localIdx i) { return op(viewA[i], viewB[i]); }
    );
}

}

template<typename ValueType>
void add(Vector<ValueType>& a, const Vector<std::type_identity_t<ValueType>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va + vb; }
    );
}

template<typename ValueType>
void sub(Vector<ValueType>& a, const Vector<std::type_identity_t<ValueType>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va - vb; }
    );
}

template<typename ValueType>
void mul(Vector<ValueType>& a, const Vector<std::type_identity_t<ValueType>>& b)
    requires requires(ValueType a, ValueType b) { a* b; }
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va * vb; }
    );
}

// operator instantiation
#define OPERATOR_INSTANTIATION(Type)                                                               \
    /* free function operator with additional requirements  */                                     \
    template void scalarMul<Type>(Vector<Type> & vector, const Type value);                        \
    template void add<Type>(Vector<Type> & a, const Vector<std::type_identity_t<Type>>& b);        \
    template void sub<Type>(Vector<Type> & a, const Vector<std::type_identity_t<Type>>& b);        \
    template void mul<Type>(Vector<Type> & a, const Vector<std::type_identity_t<Type>>& b);

#define OPERATOR_INSTANTIATION_VECT(Type)                                                          \
    /* free function operator with additional requirements  */                                     \
    template void add<Type>(Vector<Type> & a, const Vector<std::type_identity_t<Type>>& b);        \
    template void sub<Type>(Vector<Type> & a, const Vector<std::type_identity_t<Type>>& b);

OPERATOR_INSTANTIATION(uint32_t);
OPERATOR_INSTANTIATION(uint64_t);
OPERATOR_INSTANTIATION(int32_t);
OPERATOR_INSTANTIATION(int64_t);
OPERATOR_INSTANTIATION(float);
OPERATOR_INSTANTIATION(double);
OPERATOR_INSTANTIATION_VECT(Vec3);

} // namespace NeoN
