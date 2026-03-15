// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"

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
void add(Vector<ValueType>& vect, const std::type_identity_t<ValueType>& value);

/** @brief add with idx map */
template<typename ValueType>
void add(const Vector<ValueType>& in, const Vector<localIdx>& idx, Vector<ValueType>& out);

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
template<unsigned int I>
[[nodiscard]] Vector<scalar> getComponent(const Vector<Vec3>& in);

/**
 * @brief Given a Vector of Vec3 this function sets a single component
 * @returns The resulting scalar vector
 */
template<unsigned int I>
void setComponent(const Vector<scalar>& in, Vector<Vec3>& out);

/**
 * @brief Given a Vector and a set of indizes values are copied to out
 */
template<typename ValueType>
void copy(const Vector<ValueType>& in, const Vector<localIdx>& idx, Vector<ValueType>& out);

/**
 * @brief Given a Vector and a set of indizes values are copied to out
 */
template<typename ValueType>
void set(ValueType in, const Vector<localIdx>& idx, Vector<ValueType>& out);

// FIXME add test for this
/**
 * @brief Given a Vector and an index range [begin, end) a subvector is created
 */
template<typename ValueType>
Vector<ValueType> take(const Vector<ValueType>& in, localIdx first, localIdx last);

} // namespace NeoN
