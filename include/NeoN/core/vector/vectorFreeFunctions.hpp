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
 * @brief In-place variant: extract component I of @p in into the pre-existing buffer @p out,
 * reusing its storage. @p out is resized only when its size differs from @p in (so in the
 * steady-state case -- same size every call -- there is NO reallocation). Lets callers (e.g. the
 * segregated Vec3 solve) keep persistent per-component buffers instead of allocating a fresh
 * Vector every solve.
 */
template<unsigned int I>
void getComponent(const Vector<Vec3>& in, Vector<scalar>& out);

/**
 * @brief Given a Vector of Vec3 this function sets a single component
 * @returns The resulting scalar vector
 */
template<unsigned int I>
void setComponent(const Vector<scalar>& in, Vector<Vec3>& out);

/** @brief Given a Vector and an index range [first, first+length] a subvector is created
 * @returns The resulting subset vector
 */
template<typename ValueType>
Vector<ValueType> take(const Vector<ValueType>& in, std::pair<localIdx, localIdx> range);

} // namespace NeoN
