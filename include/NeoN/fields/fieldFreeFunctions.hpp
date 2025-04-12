// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <tuple>

#include <Kokkos_Core.hpp>
#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/helpers/exceptions.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/core/view.hpp"

namespace NeoFOAM
{

// Forward declaration
template<typename ValueType>
class Field;


/**
 * @brief Map a field using a specific executor.
 *
 * @param a The field to map.
 * @param inner The function to apply to each element of the field.
 * @param range The range to map the field in. If not provided, the whole field is mapped.
 */
template<typename T, typename Inner>
void map(Field<T>& a, const Inner inner, std::pair<size_t, size_t> range = {0, 0})
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = a.size();
    }
    auto viewA = a.view();
    parallelFor(
        a.exec(), {start, end}, KOKKOS_LAMBDA(const size_t i) { viewA[i] = inner(i); }
    );
}

/**
 * @brief Fill the field with a scalar value using a specific executor.
 *
 * @param field The field to fill.
 * @param value The scalar value to fill the field with.
 * @param range The range to fill the field in. If not provided, the whole field is filled.
 */
template<typename ValueType>
void fill(
    Field<ValueType>& a,
    const std::type_identity_t<ValueType> value,
    std::pair<size_t, size_t> range = {0, 0}
)
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = a.size();
    }
    auto viewA = a.view();
    parallelFor(
        a.exec(), {start, end}, KOKKOS_LAMBDA(const size_t i) { viewA[i] = value; }
    );
}


/**
 * @brief Set the field with a view of values using a specific executor.
 *
 * @param a The field to set.
 * @param b The view of values to set the field with.
 * @param range The range to set the field in. If not provided, the whole field is set.
 */
template<typename ValueType>
void setField(
    Field<ValueType>& a,
    const View<const std::type_identity_t<ValueType>> b,
    std::pair<size_t, size_t> range = {0, 0}
)
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = a.size();
    }
    auto viewA = a.view();
    parallelFor(
        a.exec(), {start, end}, KOKKOS_LAMBDA(const size_t i) { viewA[i] = b[i]; }
    );
}

template<typename ValueType>
void scalarMul(Field<ValueType>& a, const scalar value)
{
    auto viewA = a.view();
    parallelFor(
        a, KOKKOS_LAMBDA(const size_t i) { return viewA[i] * value; }
    );
}

namespace detail
{
template<typename ValueType, typename BinaryOp>
void fieldBinaryOp(
    Field<ValueType>& a, const Field<std::type_identity_t<ValueType>>& b, BinaryOp op
)
{
    NeoFOAM_ASSERT_EQUAL_LENGTH(a, b);
    auto viewA = a.view();
    auto viewB = b.view();
    parallelFor(
        a, KOKKOS_LAMBDA(const size_t i) { return op(viewA[i], viewB[i]); }
    );
}
}

template<typename ValueType>
void add(Field<ValueType>& a, const Field<std::type_identity_t<ValueType>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va + vb; }
    );
}


template<typename ValueType>
void sub(Field<ValueType>& a, const Field<std::type_identity_t<ValueType>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va - vb; }
    );
}

template<typename ValueType>
void mul(Field<ValueType>& a, const Field<std::type_identity_t<ValueType>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va * vb; }
    );
}

template<typename... Args>
auto spans(Args&... fields)
{
    return std::make_tuple(fields.view()...);
}

template<typename... Args>
auto copyToHosts(Args&... fields)
{
    return std::make_tuple(fields.copyToHost()...);
}

template<typename T>
bool equal(Field<T>& field, T value)
{
    auto hostField = field.copyToHost();
    auto hostView = hostField.view();
    for (size_t i = 0; i < hostView.size(); i++)
    {
        if (hostView[i] != value)
        {
            return false;
        }
    }
    return true;
};

template<typename T>
bool equal(const Field<T>& field, const Field<T>& field2)
{
    auto [hostField, hostField2] = copyToHosts(field, field2);
    auto [hostSpan, hostSpan2] = spans(hostField, hostField2);

    if (hostSpan.size() != hostSpan2.size())
    {
        return false;
    }

    for (size_t i = 0; i < hostSpan.size(); i++)
    {
        if (hostSpan[i] != hostSpan2[i])
        {
            return false;
        }
    }

    return true;
};

template<typename T>
bool equal(const Field<T>& field, View<T> span2)
{
    auto hostView = field.copyToHost().view();

    if (hostView.size() != span2.size())
    {
        return false;
    }

    for (size_t i = 0; i < hostView.size(); i++)
    {
        if (hostView[i] != span2[i])
        {
            return false;
        }
    }

    return true;
}

} // namespace NeoFOAM
