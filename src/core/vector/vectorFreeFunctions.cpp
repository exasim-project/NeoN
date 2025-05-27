// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include <Kokkos_Core.hpp>

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/macros.hpp"
#include "NeoN/core/view.hpp"
#include "NeoN/helpers/exceptions.hpp"

namespace NeoN
{

template<typename ValueType>
void scalarMul(Vector<ValueType>& vect, const scalar value)
    requires requires(ValueType a, scalar b) { a* b; }
{
    auto viewA = vect.view();
    parallelFor(
        vect,
        KOKKOS_LAMBDA(const localIdx i)->ValueType {
            return viewA[i] * static_cast<ValueType>(value);
        }
    );
}

template<>
void scalarMul(Vector<Vec3>& vect, const scalar value)
{
    auto viewA = vect.view();
    parallelFor(
        vect, KOKKOS_LAMBDA(const localIdx i)->Vec3 { return viewA[i] * value; }
    );
}

namespace detail
{

template<typename ValueType, typename BinaryOp>
void fieldBinaryOp(
    Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value, BinaryOp op
)
{
    auto view = vect.view();
    parallelFor(
        vect, KOKKOS_LAMBDA(const localIdx i) { return op(view[i], value); }
    );
}

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
void add(Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value)
{
    detail::fieldBinaryOp(
        vect, value, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va + vb; }
    );
}

template<typename ValueType>
void add(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2)
{
    detail::fieldBinaryOp(
        vect1, vect2, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va + vb; }
    );
}

template<typename ValueType>
void sub(Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value)
{
    detail::fieldBinaryOp(
        vect, value, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va - vb; }
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
void mul(Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value)
    requires requires(ValueType a, ValueType b) { a* b; }
{
    detail::fieldBinaryOp(
        vect, value, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va * vb; }
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
#define NN_VECTOR_OPERATOR_INSTANTIATION(Type)                                                     \
    /* free function operator with additional requirements  */                                     \
    template void scalarMul<Type>(Vector<Type>&, const scalar);                                    \
    template void add<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void add<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void sub<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void sub<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void mul<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void mul<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);

#define NN_VECTOR_OPERATOR_INSTANTIATION_VEC3(Type)                                                \
    /* free function operator with additional requirements  */                                     \
    template void scalarMul<Type>(Vector<Type>&, const scalar);                                    \
    template void add<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void add<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void sub<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void sub<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);

NN_FOR_ALL_INTEGER_TYPES(NN_VECTOR_OPERATOR_INSTANTIATION);
NN_VECTOR_OPERATOR_INSTANTIATION(float);
NN_VECTOR_OPERATOR_INSTANTIATION(double);
NN_VECTOR_OPERATOR_INSTANTIATION_VEC3(Vec3);

} // namespace NeoN
