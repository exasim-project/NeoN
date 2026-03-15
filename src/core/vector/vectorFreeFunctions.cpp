// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/core/primitives/symmTensor.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/macros.hpp"
#include "NeoN/core/view.hpp"
#include "NeoN/core/error.hpp"
#include "NeoN/helpers/exceptions.hpp"

namespace NeoN
{


template<typename ValueType>
void scalarMul(Vector<ValueType>& vect, const scalar value)
    requires requires(ValueType a, scalar b) { a* b; }
{
    if constexpr (std::is_same_v<ValueType, Vec3> || std::is_same_v<ValueType, Tensor> || std::is_same_v<ValueType, SymmTensor>)
    {
        auto viewA = vect.view();
        parallelFor(
            vect, NEON_LAMBDA(const localIdx i)->ValueType { return viewA[i] * value; }
        );
    }
    else
    {
        auto viewA = vect.view();
        parallelFor(
            vect,
            NEON_LAMBDA(const localIdx i)->ValueType {
                return viewA[i] * static_cast<ValueType>(value);
            }
        );
    }
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
        vect, NEON_LAMBDA(const localIdx i) { return op(view[i], value); }
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
        vect1, NEON_LAMBDA(const localIdx i) { return op(viewA[i], viewB[i]); }
    );
}

}

template<typename ValueType>
void add(Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value)
{
    detail::fieldBinaryOp(
        vect, value, NEON_LAMBDA(ValueType va, ValueType vb) { return va + vb; }
    );
}

template<typename ValueType>
void add(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2)
{
    detail::fieldBinaryOp(
        vect1, vect2, NEON_LAMBDA(ValueType va, ValueType vb) { return va + vb; }
    );
}

template<typename ValueType>
void add(const Vector<ValueType>& in, const Vector<localIdx>& idx, Vector<ValueType>& out)
{
    // NF_ASSERT(in.exec() == idx.exec(), "Executors are not the same");
    // NF_ASSERT(in.exec() == out.exec(), "Executors are not the same");
    // NF_ASSERT(idx.size() == out.size(), "Size mismatch");

    const auto exec = in.exec();
    const auto inV = in.view();
    const auto idxV = idx.view();
    auto outV = out.view();

    NeoN::parallelFor(
        exec, {0, idx.size()}, NEON_LAMBDA(const localIdx i) { outV[idxV[i]] = inV[i]; }, "copyMap"
    );
}

template<typename ValueType>
void sub(Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value)
{
    detail::fieldBinaryOp(
        vect, value, NEON_LAMBDA(ValueType va, ValueType vb) { return va - vb; }
    );
}

template<typename ValueType>
void sub(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2)
{
    detail::fieldBinaryOp(
        vect1, vect2, NEON_LAMBDA(ValueType va, ValueType vb) { return va - vb; }
    );
}

template<typename ValueType>
void mul(Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value)
    requires requires(ValueType a, ValueType b) { a* b; }
{
    detail::fieldBinaryOp(
        vect, value, NEON_LAMBDA(ValueType va, ValueType vb) { return va * vb; }
    );
}

template<typename ValueType>
void mul(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2)
    requires requires(ValueType a, ValueType b) { a* b; }
{
    detail::fieldBinaryOp(
        vect1, vect2, NEON_LAMBDA(ValueType va, ValueType vb) { return va * vb; }
    );
}

template<unsigned int I>
Vector<scalar> getComponent(const Vector<Vec3>& in)
{
    const auto exec = in.exec();
    const auto inV = in.view();
    auto out = Vector<scalar> {exec, in.size()};
    auto outV = out.view();

    NeoN::parallelFor(
        exec, {0, in.size()}, NEON_LAMBDA(const localIdx i) { outV[i] = inV[i][I]; }, "getVecValues"
    );
    return out;
};

template Vector<scalar> getComponent<0>(const Vector<Vec3>&);
template Vector<scalar> getComponent<1>(const Vector<Vec3>&);
template Vector<scalar> getComponent<2>(const Vector<Vec3>&);

template<unsigned int I>
void setComponent(const Vector<scalar>& in, Vector<Vec3>& out)
{
    const auto exec = in.exec();
    const auto inV = in.view();
    auto outV = out.view();

    NeoN::parallelFor(
        exec, {0, in.size()}, NEON_LAMBDA(const localIdx i) { outV[i][I] = inV[i]; }, "setVecValues"
    );
};

template void setComponent<0>(const Vector<scalar>&, Vector<Vec3>&);
template void setComponent<1>(const Vector<scalar>&, Vector<Vec3>&);
template void setComponent<2>(const Vector<scalar>&, Vector<Vec3>&);

template<typename ValueType>
void copy(const Vector<ValueType>& in, const Vector<localIdx>& idx, Vector<ValueType>& out)
{
    NF_ASSERT(in.exec() == idx.exec(), "Executors are not the same");
    NF_ASSERT(in.exec() == out.exec(), "Executors are not the same");
    NF_ASSERT(idx.size() == out.size(), "Size mismatch");

    const auto exec = in.exec();
    const auto inV = in.view();
    const auto idxV = idx.view();
    auto outV = out.view();

    NeoN::parallelFor(
        exec, {0, idx.size()}, NEON_LAMBDA(const localIdx i) { outV[i] = inV[idxV[i]]; }, "copyMap"
    );
};

template<typename ValueType>
void set(ValueType in, const Vector<localIdx>& idx, Vector<ValueType>& out)
{

    const auto exec = out.exec();
    const auto idxV = idx.view();
    auto outV = out.view();

    NeoN::parallelFor(
        exec, {0, idx.size()}, NEON_LAMBDA(const localIdx i) { outV[idxV[i]] = in; }, "copyMap"
    );
}

template<typename ValueType>
Vector<ValueType> take(const Vector<ValueType>& in, localIdx first, localIdx last)
{
    NF_ASSERT(last >= first, "Invalid index range");
    const auto exec = in.exec();

    auto out = Vector<ValueType>(exec, (last - first));
    auto outV = out.view();
    const auto inV = in.view();

    NeoN::parallelFor(
        exec, {first, last}, NEON_LAMBDA(const localIdx i) { outV[i - first] = inV[i]; }, "copyMap"
    );

    return out;
}

// operator instantiation
#define NN_VECTOR_OPERATOR_INSTANTIATION(Type)                                                     \
    /* free function operator with additional requirements  */                                     \
    template void copy<Type>(const Vector<Type>&, const Vector<localIdx>&, Vector<Type>&);         \
    template void set<Type>(Type, const Vector<localIdx>&, Vector<Type>&);                         \
    template Vector<Type> take<Type>(const Vector<Type>&, localIdx, localIdx);                     \
    template void scalarMul<Type>(Vector<Type>&, const scalar);                                    \
    template void add<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void add<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void add(const Vector<Type>&, const Vector<localIdx>&, Vector<Type>&);                \
    template void sub<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void sub<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void mul<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void mul<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);

#define NN_VECTOR_OPERATOR_INSTANTIATION_VEC3(Type)                                                \
    /* free function operator with additional requirements  */                                     \
    template void scalarMul<Type>(Vector<Type>&, const scalar);                                    \
    template Vector<Type> take<Type>(const Vector<Type>&, localIdx, localIdx);                     \
    template void add<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void add<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void sub<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void sub<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void mul<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void mul<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);

NN_FOR_ALL_INTEGER_TYPES(NN_VECTOR_OPERATOR_INSTANTIATION);
NN_VECTOR_OPERATOR_INSTANTIATION(float);
NN_VECTOR_OPERATOR_INSTANTIATION(double);
NN_VECTOR_OPERATOR_INSTANTIATION_VEC3(Vec3);

// Tensor types support +/-/scalarMul but not element-wise mul(Tensor,Tensor)
#define NN_VECTOR_OPERATOR_INSTANTIATION_TENSOR(Type)                                              \
    template void scalarMul<Type>(Vector<Type>&, const scalar);                                    \
    template void add<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void add<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);             \
    template void sub<Type>(Vector<Type>&, const std::type_identity_t<Type>&);                     \
    template void sub<Type>(Vector<Type>&, const Vector<std::type_identity_t<Type>>&);

NN_VECTOR_OPERATOR_INSTANTIATION_TENSOR(Tensor);
NN_VECTOR_OPERATOR_INSTANTIATION_TENSOR(SymmTensor);

} // namespace NeoN
