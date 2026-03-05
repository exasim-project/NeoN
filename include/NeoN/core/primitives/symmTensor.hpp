// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp> // IWYU pragma: keep

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/traits.hpp"


namespace NeoN
{


/**
 * @class SymmTensor
 * @brief A class for the representation of a symmetric 3x3 tensor (upper triangle storage)
 * @ingroup primitives
 *
 * Components stored as: xx, xy, xz, yy, yz, zz
 */
class SymmTensor
{
public:

    KOKKOS_INLINE_FUNCTION
    SymmTensor()
    {
        for (size_t i = 0; i < 6; i++)
        {
            cmpts_[i] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor(scalar xx, scalar xy, scalar xz, scalar yy, scalar yz, scalar zz)
    {
        cmpts_[0] = xx;
        cmpts_[1] = xy;
        cmpts_[2] = xz;
        cmpts_[3] = yy;
        cmpts_[4] = yz;
        cmpts_[5] = zz;
    }

    KOKKOS_INLINE_FUNCTION
    explicit SymmTensor(const scalar constValue)
    {
        for (size_t i = 0; i < 6; i++)
        {
            cmpts_[i] = constValue;
        }
    }

    scalar* data() { return cmpts_; }

    const scalar* data() const { return cmpts_; }

    constexpr size_t size() const { return 6; }

    KOKKOS_INLINE_FUNCTION
    scalar& operator[](const size_t i) { return cmpts_[i]; }

    KOKKOS_INLINE_FUNCTION
    scalar operator[](const size_t i) const { return cmpts_[i]; }

    // Named component accessors
    KOKKOS_INLINE_FUNCTION scalar xx() const { return cmpts_[0]; }

    KOKKOS_INLINE_FUNCTION scalar xy() const { return cmpts_[1]; }

    KOKKOS_INLINE_FUNCTION scalar xz() const { return cmpts_[2]; }

    KOKKOS_INLINE_FUNCTION scalar yy() const { return cmpts_[3]; }

    KOKKOS_INLINE_FUNCTION scalar yz() const { return cmpts_[4]; }

    KOKKOS_INLINE_FUNCTION scalar zz() const { return cmpts_[5]; }

    KOKKOS_INLINE_FUNCTION
    bool operator==(const SymmTensor& rhs) const
    {
        for (size_t i = 0; i < 6; i++)
        {
            if (cmpts_[i] != rhs.cmpts_[i]) return false;
        }
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor operator+(const SymmTensor& rhs) const
    {
        SymmTensor result;
        for (size_t i = 0; i < 6; i++)
        {
            result.cmpts_[i] = cmpts_[i] + rhs.cmpts_[i];
        }
        return result;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor& operator+=(const SymmTensor& rhs)
    {
        for (size_t i = 0; i < 6; i++)
        {
            cmpts_[i] += rhs.cmpts_[i];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor operator-(const SymmTensor& rhs) const
    {
        SymmTensor result;
        for (size_t i = 0; i < 6; i++)
        {
            result.cmpts_[i] = cmpts_[i] - rhs.cmpts_[i];
        }
        return result;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor& operator-=(const SymmTensor& rhs)
    {
        for (size_t i = 0; i < 6; i++)
        {
            cmpts_[i] -= rhs.cmpts_[i];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor operator*(const scalar& rhs) const
    {
        SymmTensor result;
        for (size_t i = 0; i < 6; i++)
        {
            result.cmpts_[i] = cmpts_[i] * rhs;
        }
        return result;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor& operator*=(const scalar& rhs)
    {
        for (size_t i = 0; i < 6; i++)
        {
            cmpts_[i] *= rhs;
        }
        return *this;
    }

private:

    scalar cmpts_[6];
};


KOKKOS_INLINE_FUNCTION
SymmTensor operator*(const scalar& sclr, SymmTensor rhs)
{
    rhs *= sclr;
    return rhs;
}

KOKKOS_INLINE_FUNCTION
SymmTensor operator/(const SymmTensor& lhs, scalar rhs)
{
    SymmTensor result;
    for (size_t i = 0; i < 6; i++)
    {
        result[i] = lhs[i] / rhs;
    }
    return result;
}

KOKKOS_INLINE_FUNCTION
scalar mag(const SymmTensor& s)
{
    // Frobenius norm: diagonal terms once, off-diagonal terms twice (symmetric)
    return sqrt(
        s.xx() * s.xx() + s.yy() * s.yy() + s.zz() * s.zz() + 2.0 * s.xy() * s.xy()
        + 2.0 * s.xz() * s.xz() + 2.0 * s.yz() * s.yz()
    );
}

KOKKOS_INLINE_FUNCTION
SymmTensor dev(const SymmTensor& s)
{
    scalar tr = (s.xx() + s.yy() + s.zz()) / 3.0;
    return SymmTensor(s.xx() - tr, s.xy(), s.xz(), s.yy() - tr, s.yz(), s.zz() - tr);
}

std::ostream& operator<<(std::ostream& out, const SymmTensor& s);


template<>
KOKKOS_INLINE_FUNCTION SymmTensor one<SymmTensor>()
{
    return SymmTensor(1.0);
}

template<>
KOKKOS_INLINE_FUNCTION SymmTensor zero<SymmTensor>()
{
    return SymmTensor(0.0);
}

template<>
KOKKOS_INLINE_FUNCTION SymmTensor inv<SymmTensor>(SymmTensor in)
{
    SymmTensor result;
    for (size_t i = 0; i < 6; i++)
    {
        result[i] = 1.0 / in[i];
    }
    return result;
}


} // namespace NeoN
