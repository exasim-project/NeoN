// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/vector/vector.hpp"
#ifdef USE_JULIA
#include <julia.h>
#endif
namespace NeoN::dsl
{

/**
 * @class Coeff
 * @brief A class that represents a coefficient for the NeoN dsl.
 *
 * This class stores a single scalar coefficient and optionally view of values.
 * It is used to delay the evaluation of a scalar multiplication with a field to
 * avoid the creation of a temporary field copy.
 * It provides an indexing operator `operator[]` that returns the evaluated value at the specified
 * index.
 */
class Coeff
{

public:

    Coeff();

    Coeff(scalar value);

    Coeff(scalar coeff, const Vector<scalar>& field);

    Coeff(const Vector<scalar>& field);

    KOKKOS_INLINE_FUNCTION
    scalar operator[](const localIdx i) const { return (hasView_) ? view_[i] * coeff_ : coeff_; }

    bool hasView();

    View<const scalar> view();

    Coeff& operator*=(scalar rhs);


    Coeff& operator*=(const Coeff& rhs);
#ifdef USE_JULIA

    // jl_array_t* juliaPtr() const
    // {
    //     if (hasView_)
    //     {
    //         if constexpr (std::is_same_v<scalar, float>)
    //         {
    //             jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float32_type, 1);
    //             jl_array_t* julia_ptr = jl_ptr_to_array_1d(array_type, view_.data, view_.size(),
    //             0); return julia_ptr;
    //         }
    //         else if constexpr (std::is_same_v<scalar, double>)
    //         {
    //             jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    //             jl_array_t* julia_ptr = jl_ptr_to_array_1d(array_type, view_.data, view_.size(),
    //             0); return julia_ptr;
    //         }
    //     }
    //     else
    //     {
    //         std::cerr << "Coeff has no view\n"
    //     }
    // }
#endif
private:

    scalar coeff_;

    View<const scalar> view_;

    bool hasView_;
};


[[nodiscard]] inline Coeff operator*(const Coeff& lhs, const Coeff& rhs)
{
    Coeff result = lhs;
    result *= rhs;
    return result;
}

namespace detail
{
/* @brief function to force evaluation to a field, the field will be resized to hold either a
 * single value or the full field
 *
 * @param field to store the result
 */
void toVector(Coeff& coeff, Vector<scalar>& rhs);

} // namespace detail

} // namespace NeoN::dsl
