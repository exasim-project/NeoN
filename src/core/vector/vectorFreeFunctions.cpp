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
    Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2, BinaryOp op
)
{
    NeoN_ASSERT_EQUAL_LENGTH(vect1, vect2);
    auto viewA = vect1.view();
    auto viewB = vect2.view();
    parallelFor(
        vect1, KOKKOS_LAMBDA(const localIdx i) { return op(viewA[i], viewB[i]); }
    );
}

}

template<typename ValueType>
void add(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2)
{
    detail::fieldBinaryOp(
        vect1, vect2, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va + vb; }
    );
}

template<typename ValueType>
void sub(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2)
{
    detail::fieldBinaryOp(
        vect1, vect2, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va - vb; }
    );
}

template<typename ValueType>
void mul(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2)
    requires requires(ValueType a, ValueType b) { a* b; }
{
    detail::fieldBinaryOp(
        vect1, vect2, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va * vb; }
    );
}

// operator instantiation
#define OPERATOR_INSTANTIATION(Type)                                                               \
    /* free function operator with additional requirements  */                                     \
    template void scalarMul<Type>(Vector<Type> & vector, const Type value);                        \
    template void add<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void sub<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void mul<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);

#define OPERATOR_INSTANTIATION_VECT(Type)                                                          \
    /* free function operator with additional requirements  */                                     \
    template void add<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void sub<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);

OPERATOR_INSTANTIATION(uint32_t);
OPERATOR_INSTANTIATION(uint64_t);
OPERATOR_INSTANTIATION(int32_t);
OPERATOR_INSTANTIATION(int64_t);
OPERATOR_INSTANTIATION(float);
OPERATOR_INSTANTIATION(double);
OPERATOR_INSTANTIATION_VECT(Vec3);

} // namespace NeoN
