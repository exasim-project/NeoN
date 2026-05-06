// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"

#include <type_traits>
#ifdef USE_JULIA
#include <julia.h>
#endif

namespace NeoN
{

template<typename ValueType>
class Vector;

#ifdef USE_JULIA
template<typename ValueType>
jl_array_t* transposeToJulia(Vector<ValueType>& vect)
{
    if constexpr (std::is_same_v<ValueType, Vec3>)
    {
        auto viewA = vect.view();
        jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 2);

        size_t dims[2] = {3, vect.size()};

        jl_array_t* arr = jl_alloc_array_nd(array_type, dims, 2);

        float* p = jl_array_data(arr, float);

        for (size_t i = 0; i < vect.size(); ++i)
        {
            p[0 + 3 * i] = viewA[i][0];
            p[1 + 3 * i] = viewA[i][1];
            p[2 + 3 * i] = viewA[i][2];
        }
        return arr;
    }
    else
    {
        // std::cout << "no transpose needed, calling juliaPtr!\n";
        return vect.juliaPtr();
    }
};
#endif

template<typename ValueType>
void scalarMul(Vector<ValueType>& vect, const scalar value)
    requires requires(ValueType a, scalar b) { a * b; };

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
    requires requires(ValueType a, ValueType b) { a * b; };

template<typename ValueType>
void mul(Vector<ValueType>& vect1, const Vector<std::type_identity_t<ValueType>>& vect2)
    requires requires(ValueType a, ValueType b) { a * b; };

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
} // namespace NeoN
