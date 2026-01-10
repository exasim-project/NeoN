// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"

#include <type_traits>

namespace NeoN
{

template<typename ValueType>
class Vector;

template<typename ValueType>
void scalarMul(Vector<ValueType>& vect, const scalar value)
    requires requires(ValueType a, scalar b) { a* b; };

namespace detail
{

template<typename ValueType, typename BinaryOp>
void fieldBinaryOp(
    Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2, BinaryOp op
);

}

template<typename ValueType>
void add(Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value);

template<typename ValueType>
void add(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2);

template<typename ValueType>
void sub(Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value);

template<typename ValueType>
void sub(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2);

template<typename ValueType>
void mul(Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value)
    requires requires(ValueType a, ValueType b) { a* b; };

template<typename ValueType>
void mul(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2)
    requires requires(ValueType a, ValueType b) { a* b; };

/**
 * @brief Given a Vector of Vec3 this function extracts a single component
 * @returns The resulting scalar vector
 */
template<unsigned int I, typename VectorType>
[[nodiscard]] Vector<scalar> get(const Vector<VectorType>& in);
} // namespace NeoN
