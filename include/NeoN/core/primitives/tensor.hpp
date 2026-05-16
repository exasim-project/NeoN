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

class SymmTensor; // forward declaration

/**
 * @class Tensor
 * @brief A class for the representation of a 3x3 tensor (row-major)
 * @ingroup primitives
 */
class Tensor
{
public:

    KOKKOS_INLINE_FUNCTION
    Tensor()
    {
        for (size_t i = 0; i < 9; i++)
        {
            cmpts_[i] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    Tensor(
        scalar xx,
        scalar xy,
        scalar xz,
        scalar yx,
        scalar yy,
        scalar yz,
        scalar zx,
        scalar zy,
        scalar zz
    )
    {
        cmpts_[0] = xx;
        cmpts_[1] = xy;
        cmpts_[2] = xz;
        cmpts_[3] = yx;
        cmpts_[4] = yy;
        cmpts_[5] = yz;
        cmpts_[6] = zx;
        cmpts_[7] = zy;
        cmpts_[8] = zz;
    }

    KOKKOS_INLINE_FUNCTION
    explicit Tensor(const scalar constValue)
    {
        for (size_t i = 0; i < 9; i++)
        {
            cmpts_[i] = constValue;
        }
    }

    scalar* data() { return cmpts_; }

    const scalar* data() const { return cmpts_; }

    constexpr size_t size() const { return 9; }

    KOKKOS_INLINE_FUNCTION
    scalar& operator[](const size_t i) { return cmpts_[i]; }

    KOKKOS_INLINE_FUNCTION
    scalar operator[](const size_t i) const { return cmpts_[i]; }

    KOKKOS_INLINE_FUNCTION
    scalar& operator()(const size_t row, const size_t col) { return cmpts_[row * 3 + col]; }

    KOKKOS_INLINE_FUNCTION
    scalar operator()(const size_t row, const size_t col) const { return cmpts_[row * 3 + col]; }

    // Named component accessors
    KOKKOS_INLINE_FUNCTION scalar xx() const { return cmpts_[0]; }

    KOKKOS_INLINE_FUNCTION scalar xy() const { return cmpts_[1]; }

    KOKKOS_INLINE_FUNCTION scalar xz() const { return cmpts_[2]; }

    KOKKOS_INLINE_FUNCTION scalar yx() const { return cmpts_[3]; }

    KOKKOS_INLINE_FUNCTION scalar yy() const { return cmpts_[4]; }

    KOKKOS_INLINE_FUNCTION scalar yz() const { return cmpts_[5]; }

    KOKKOS_INLINE_FUNCTION scalar zx() const { return cmpts_[6]; }

    KOKKOS_INLINE_FUNCTION scalar zy() const { return cmpts_[7]; }

    KOKKOS_INLINE_FUNCTION scalar zz() const { return cmpts_[8]; }

    KOKKOS_INLINE_FUNCTION
    bool operator==(const Tensor& rhs) const
    {
        for (size_t i = 0; i < 9; i++)
        {
            if (cmpts_[i] != rhs.cmpts_[i]) return false;
        }
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor operator+(const Tensor& rhs) const
    {
        Tensor result;
        for (size_t i = 0; i < 9; i++)
        {
            result.cmpts_[i] = cmpts_[i] + rhs.cmpts_[i];
        }
        return result;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor& operator+=(const Tensor& rhs)
    {
        for (size_t i = 0; i < 9; i++)
        {
            cmpts_[i] += rhs.cmpts_[i];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor operator-(const Tensor& rhs) const
    {
        Tensor result;
        for (size_t i = 0; i < 9; i++)
        {
            result.cmpts_[i] = cmpts_[i] - rhs.cmpts_[i];
        }
        return result;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor& operator-=(const Tensor& rhs)
    {
        for (size_t i = 0; i < 9; i++)
        {
            cmpts_[i] -= rhs.cmpts_[i];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor operator*(const scalar& rhs) const
    {
        Tensor result;
        for (size_t i = 0; i < 9; i++)
        {
            result.cmpts_[i] = cmpts_[i] * rhs;
        }
        return result;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor& operator*=(const scalar& rhs)
    {
        for (size_t i = 0; i < 9; i++)
        {
            cmpts_[i] *= rhs;
        }
        return *this;
    }

private:

    scalar cmpts_[9];
};


KOKKOS_INLINE_FUNCTION
Tensor operator*(const scalar& sclr, Tensor rhs)
{
    rhs *= sclr;
    return rhs;
}

KOKKOS_INLINE_FUNCTION
Tensor operator/(const Tensor& lhs, scalar rhs)
{
    Tensor result;
    for (size_t i = 0; i < 9; i++)
    {
        result[i] = lhs[i] / rhs;
    }
    return result;
}

KOKKOS_INLINE_FUNCTION
scalar mag(const Tensor& t)
{
    scalar sumSq = 0.0;
    for (size_t i = 0; i < 9; i++)
    {
        sumSq += t[i] * t[i];
    }
    return sqrt(sumSq);
}

KOKKOS_INLINE_FUNCTION
Tensor T(const Tensor& t)
{
    return Tensor(t.xx(), t.yx(), t.zx(), t.xy(), t.yy(), t.zy(), t.xz(), t.yz(), t.zz());
}

KOKKOS_INLINE_FUNCTION
Tensor skew(const Tensor& t)
{
    Tensor tT = T(t);
    return (t - tT) * 0.5;
}

/** @brief Deviatoric part (2/3 variant): dev2(T) = T - (2/3)*tr(T)*I */
KOKKOS_INLINE_FUNCTION
Tensor dev2(const Tensor& t)
{
    scalar tr23 = (2.0 / 3.0) * (t.xx() + t.yy() + t.zz());
    return Tensor(
        t.xx() - tr23, t.xy(), t.xz(), t.yx(), t.yy() - tr23, t.yz(), t.zx(), t.zy(), t.zz() - tr23
    );
}

std::ostream& operator<<(std::ostream& out, const Tensor& t);


template<>
KOKKOS_INLINE_FUNCTION Tensor one<Tensor>()
{
    return Tensor(1.0);
}

template<>
KOKKOS_INLINE_FUNCTION Tensor zero<Tensor>()
{
    return Tensor(0.0);
}

template<>
KOKKOS_INLINE_FUNCTION Tensor inv<Tensor>(Tensor in)
{
    Tensor result;
    for (size_t i = 0; i < 9; i++)
    {
        result[i] = 1.0 / in[i];
    }
    return result;
}


} // namespace NeoN

// Cross-type functions requiring SymmTensor and Vec3 definitions
#include "NeoN/core/primitives/symmTensor.hpp"
#include "NeoN/core/primitives/vec3.hpp"

namespace NeoN
{

/** @brief Symmetric part: 0.5*(T + T^T) */
KOKKOS_INLINE_FUNCTION
SymmTensor symm(const Tensor& t)
{
    return SymmTensor(
        t.xx(),
        0.5 * (t.xy() + t.yx()),
        0.5 * (t.xz() + t.zx()),
        t.yy(),
        0.5 * (t.yz() + t.zy()),
        t.zz()
    );
}

/** @brief Twice the symmetric part: T + T^T */
KOKKOS_INLINE_FUNCTION
SymmTensor twoSymm(const Tensor& t)
{
    return SymmTensor(
        2.0 * t.xx(), t.xy() + t.yx(), t.xz() + t.zx(), 2.0 * t.yy(), t.yz() + t.zy(), 2.0 * t.zz()
    );
}

/** @brief Inner product v · T → Vec3 (row contraction: result_i = Σ_j v_j * T_ji) */
KOKKOS_INLINE_FUNCTION
Vec3 inner(const Vec3& v, const Tensor& t)
{
    return Vec3(
        v[0] * t.xx() + v[1] * t.yx() + v[2] * t.zx(),
        v[0] * t.xy() + v[1] * t.yy() + v[2] * t.zy(),
        v[0] * t.xz() + v[1] * t.yz() + v[2] * t.zz()
    );
}

/** @brief Trace of a tensor: tr(T) = T_xx + T_yy + T_zz */
KOKKOS_INLINE_FUNCTION
scalar tr(const Tensor& t) { return t.xx() + t.yy() + t.zz(); }

/** @brief devTwoSymm(T) = dev(twoSymm(T)) — deviatoric of twice the symmetric part.
 *  twoSymm(T) = T + T^T, tr(twoSymm) = 2*tr(T)
 *  dev(S) = S - (1/3)*tr(S)*I → dev(twoSymm(T)) = twoSymm(T) - (2/3)*tr(T)*I */
KOKKOS_INLINE_FUNCTION
SymmTensor devTwoSymm(const Tensor& t)
{
    scalar tr23 = (2.0 / 3.0) * (t.xx() + t.yy() + t.zz());
    return SymmTensor(
        2.0 * t.xx() - tr23,
        t.xy() + t.yx(),
        t.xz() + t.zx(),
        2.0 * t.yy() - tr23,
        t.yz() + t.zy(),
        2.0 * t.zz() - tr23
    );
}

/** @brief Double inner product T:S → scalar (Frobenius: Σ_ij T_ij * S_ij) */
KOKKOS_INLINE_FUNCTION
scalar doubleInner(const Tensor& t, const SymmTensor& s)
{
    return t.xx() * s.xx() + t.xy() * s.xy() + t.xz() * s.xz() + t.yx() * s.xy() + t.yy() * s.yy()
         + t.yz() * s.yz() + t.zx() * s.xz() + t.zy() * s.yz() + t.zz() * s.zz();
}

} // namespace NeoN
