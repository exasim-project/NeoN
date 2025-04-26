// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <tuple>

#include <Kokkos_Core.hpp>
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/helpers/exceptions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/view.hpp"

namespace NeoN
{

// Forward declaration
template<typename ValueType>
class Vector;


/**
 * @brief Map a field using a specific executor.
 *
 * @param a The field to map.
 * @param inner The function to apply to each element of the field.
 * @param range The range to map the field in. If not provided, the whole field is mapped.
 */
template<typename ValueType, typename Inner>
void map(Vector<ValueType>& a, const Inner inner, std::pair<localIdx, localIdx> range = {0, 0})
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = a.size();
    }
    auto viewA = a.view();
    parallelFor(
        a.exec(), {start, end}, KOKKOS_LAMBDA(const localIdx i) { viewA[i] = inner(i); }
    );
}

/**
 * @brief Fill the field with a vector value using a specific executor.
 *
 * @param field The field to fill.
 * @param value The vector value to fill the field with.
 * @param range The range to fill the field in. If not provided, the whole field is filled.
 */
template<typename ValueType>
void fill(
    Vector<ValueType>& a,
    const std::type_identity_t<ValueType> value,
    std::pair<localIdx, localIdx> range = {0, 0}
)
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = a.size();
    }
    auto viewA = a.view();
    parallelFor(
        a.exec(), {start, end}, KOKKOS_LAMBDA(const localIdx i) { viewA[i] = value; }
    );
}


/**
 * @brief Set the vector with a view of values using a specific executor.
 *
 * @param a The vector to set.
 * @param b The view of values to set the vector with.
 * @param range The range to set the vector in. If not provided, the whole vector is set.
 */
template<typename ValueType>
void setVector(
    Vector<ValueType>& a,
    const View<const std::type_identity_t<ValueType>> b,
    std::pair<localIdx, localIdx> range = {0, 0}
)
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = a.size();
    }
    auto viewA = a.view();
    parallelFor(
        a.exec(), {start, end}, KOKKOS_LAMBDA(const localIdx i) { viewA[i] = b[i]; }
    );
}

template<typename ValueType>
void scalarMul(Vector<ValueType>& a, const ValueType value)
{
    auto viewA = a.view();
    parallelFor(
        a, KOKKOS_LAMBDA(const localIdx i) { return viewA[i] * value; }
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
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va * vb; }
    );
}

template<typename... Args>
auto copyToHosts(Args&... fields)
{
    return std::make_tuple(fields.copyToHost()...);
}

template<typename ValueType>
bool equal(Vector<ValueType>& field, ValueType value)
{
    auto hostVector = field.copyToHost();
    auto hostView = hostVector.view();
    for (localIdx i = 0; i < hostView.size(); i++)
    {
        if (hostView[i] != value)
        {
            return false;
        }
    }
    return true;
};

template<typename ValueType>
bool equal(const Vector<ValueType>& field, const Vector<ValueType>& field2)
{
    auto [hostVector, hostVector2] = copyToHosts(field, field2);
    auto [hostView, hostView2] = views(hostVector, hostVector2);

    if (hostView.size() != hostView2.size())
    {
        return false;
    }

    for (localIdx i = 0; i < hostView.size(); i++)
    {
        if (hostView[i] != hostView2[i])
        {
            return false;
        }
    }

    return true;
};

template<typename ValueType>
bool equal(const Vector<ValueType>& field, View<ValueType> view2)
{
    auto hostView = field.copyToHost().view();

    if (hostView.size() != view2.size())
    {
        return false;
    }

    for (localIdx i = 0; i < hostView.size(); i++)
    {
        if (hostView[i] != view2[i])
        {
            return false;
        }
    }

    return true;
}

} // namespace NeoN
