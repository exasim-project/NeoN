// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include <Kokkos_Core.hpp> // IWYU pragma: keep

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/traits.hpp"


namespace NeoN
{


/**
 * @class Vector
 * @brief A class for the representation of a 3D Vector
 * @ingroup primitives
 */
class Vector
{
public:

    KOKKOS_INLINE_FUNCTION
    Vector()
    {
        cmpts_[0] = 0.0;
        cmpts_[1] = 0.0;
        cmpts_[2] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    Vector(scalar x, scalar y, scalar z)
    {
        cmpts_[0] = x;
        cmpts_[1] = y;
        cmpts_[2] = z;
    }

    KOKKOS_INLINE_FUNCTION
    explicit Vector(const scalar constValue)
    {
        cmpts_[0] = constValue;
        cmpts_[1] = constValue;
        cmpts_[2] = constValue;
    }

    /**
     * @brief Returns pointer to the data of the vector
     *
     * @return point to the first scalar
     */
    scalar* data() { return cmpts_; }

    /**
     * @brief Returns pointer to the data of the vector
     *
     * @return point to the first scalar
     */
    const scalar* data() const { return cmpts_; }

    /**
     * @brief Returns the size of the vector
     *
     * @return The size of the vector
     */
    constexpr size_t size() const { return 3; }

    KOKKOS_INLINE_FUNCTION
    scalar& operator[](const size_t i) { return cmpts_[i]; }

    KOKKOS_INLINE_FUNCTION
    scalar operator[](const size_t i) const { return cmpts_[i]; }

    KOKKOS_INLINE_FUNCTION
    scalar& operator()(const size_t i) { return cmpts_[i]; }

    KOKKOS_INLINE_FUNCTION
    scalar operator()(const size_t i) const { return cmpts_[i]; }

    KOKKOS_INLINE_FUNCTION
    bool operator==(const Vector& rhs) const
    {
        return cmpts_[0] == rhs(0) && cmpts_[1] == rhs(1) && cmpts_[2] == rhs(2);
    }

    KOKKOS_INLINE_FUNCTION
    Vector operator+(const Vector& rhs) const
    {
        return Vector(cmpts_[0] + rhs(0), cmpts_[1] + rhs(1), cmpts_[2] + rhs(2));
    }

    KOKKOS_INLINE_FUNCTION
    Vector& operator+=(const Vector& rhs)
    {
        cmpts_[0] += rhs(0);
        cmpts_[1] += rhs(1);
        cmpts_[2] += rhs(2);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Vector operator-(const Vector& rhs) const
    {
        return Vector(cmpts_[0] - rhs(0), cmpts_[1] - rhs(1), cmpts_[2] - rhs(2));
    }

    KOKKOS_INLINE_FUNCTION
    Vector& operator-=(const Vector& rhs)
    {
        cmpts_[0] -= rhs(0);
        cmpts_[1] -= rhs(1);
        cmpts_[2] -= rhs(2);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Vector operator*(const scalar& rhs) const
    {
        return Vector(cmpts_[0] * rhs, cmpts_[1] * rhs, cmpts_[2] * rhs);
    }


    KOKKOS_INLINE_FUNCTION
    Vector operator*(const label& rhs) const
    {
        return Vector(cmpts_[0] * rhs, cmpts_[1] * rhs, cmpts_[2] * rhs);
    }


    KOKKOS_INLINE_FUNCTION
    Vector& operator*=(const scalar& rhs)
    {
        cmpts_[0] *= rhs;
        cmpts_[1] *= rhs;
        cmpts_[2] *= rhs;
        return *this;
    }

private:

    scalar cmpts_[3];
};


KOKKOS_INLINE_FUNCTION
Vector operator*(const scalar& sclr, Vector rhs)
{
    rhs *= sclr;
    return rhs;
}

KOKKOS_INLINE_FUNCTION
scalar operator&(const Vector& lhs, Vector rhs)
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

KOKKOS_INLINE_FUNCTION
scalar mag(const Vector& vec) { return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]); }

std::ostream& operator<<(std::ostream& out, const Vector& vec);


template<>
KOKKOS_INLINE_FUNCTION Vector one<Vector>()
{
    return Vector(1.0, 1.0, 1.0);
}

template<>
KOKKOS_INLINE_FUNCTION Vector zero<Vector>()
{
    return Vector(0.0, 0.0, 0.0);
}

} // namespace NeoN
