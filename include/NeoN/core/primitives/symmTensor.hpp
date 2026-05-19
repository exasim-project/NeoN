// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp> // IWYU pragma: keep

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/core/primitives/traits.hpp"


namespace NeoN
{


/**
 * @class SymmTensor
 * @brief A class for the representation of a symmetric 3x3 tensor
 * @ingroup primitives
 *
 * Upper-triangle storage order: [xx, xy, xz, yy, yz, zz]
 * i.e. data_[0]=xx, data_[1]=xy, data_[2]=xz, data_[3]=yy, data_[4]=yz, data_[5]=zz.
 */
class SymmTensor
{
public:

    KOKKOS_INLINE_FUNCTION
    SymmTensor()
    {
        for (int k = 0; k < 6; ++k)
        {
            data_[k] = 0.0;
        }
    }

    /** @brief Construct scalar-times-identity symmetric tensor */
    KOKKOS_INLINE_FUNCTION
    explicit SymmTensor(const scalar diag)
    {
        data_[0] = diag; // xx
        data_[1] = 0.0;  // xy
        data_[2] = 0.0;  // xz
        data_[3] = diag; // yy
        data_[4] = 0.0;  // yz
        data_[5] = diag; // zz
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor(scalar xx, scalar xy, scalar xz, scalar yy, scalar yz, scalar zz)
    {
        data_[0] = xx;
        data_[1] = xy;
        data_[2] = xz;
        data_[3] = yy;
        data_[4] = yz;
        data_[5] = zz;
    }

    KOKKOS_INLINE_FUNCTION scalar* data() { return data_; }
    KOKKOS_INLINE_FUNCTION const scalar* data() const { return data_; }

    KOKKOS_INLINE_FUNCTION constexpr size_t size() const { return 6; }

    /** @brief Component access — maps (i,j) to the stored upper-triangle */
    KOKKOS_INLINE_FUNCTION
    scalar operator()(const size_t i, const size_t j) const
    {
        // index map: (0,0)->0, (0,1)->1, (0,2)->2, (1,1)->3, (1,2)->4, (2,2)->5
        // symmetric: (1,0)->(0,1), (2,0)->(0,2), (2,1)->(1,2)
        static constexpr int lut[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        return data_[lut[i][j]];
    }

    KOKKOS_INLINE_FUNCTION
    scalar& operator()(const size_t i, const size_t j)
    {
        static constexpr int lut[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        return data_[lut[i][j]];
    }

    KOKKOS_INLINE_FUNCTION
    bool operator==(const SymmTensor& rhs) const
    {
        for (int k = 0; k < 6; ++k)
        {
            if (data_[k] != rhs.data_[k]) return false;
        }
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor operator+(const SymmTensor& rhs) const
    {
        SymmTensor res;
        for (int k = 0; k < 6; ++k)
        {
            res.data_[k] = data_[k] + rhs.data_[k];
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor& operator+=(const SymmTensor& rhs)
    {
        for (int k = 0; k < 6; ++k)
        {
            data_[k] += rhs.data_[k];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor operator-(const SymmTensor& rhs) const
    {
        SymmTensor res;
        for (int k = 0; k < 6; ++k)
        {
            res.data_[k] = data_[k] - rhs.data_[k];
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor& operator-=(const SymmTensor& rhs)
    {
        for (int k = 0; k < 6; ++k)
        {
            data_[k] -= rhs.data_[k];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor operator*(const scalar s) const
    {
        SymmTensor res;
        for (int k = 0; k < 6; ++k)
        {
            res.data_[k] = data_[k] * s;
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    SymmTensor& operator*=(const scalar s)
    {
        for (int k = 0; k < 6; ++k)
        {
            data_[k] *= s;
        }
        return *this;
    }

    /** @brief Trace: xx + yy + zz */
    KOKKOS_INLINE_FUNCTION
    scalar trace() const { return data_[0] + data_[3] + data_[5]; }

    /** @brief Deviatoric part: S - (1/3)*tr(S)*I */
    KOKKOS_INLINE_FUNCTION
    SymmTensor dev() const
    {
        const scalar tr3 = trace() / scalar(3);
        return SymmTensor(
            data_[0] - tr3, data_[1], data_[2], data_[3] - tr3, data_[4], data_[5] - tr3
        );
    }

    /** @brief dev2: S - (2/3)*tr(S)*I  (used in viscous stress) */
    KOKKOS_INLINE_FUNCTION
    SymmTensor dev2() const
    {
        // TODO in case of trace returning a float, when NeoN_DEFINE_DP_SCALAR = OFF, this will
        // trigger a compiler warning
        const scalar tr23 = (2.0 / 3.0) * trace();
        return SymmTensor(
            data_[0] - tr23, data_[1], data_[2], data_[3] - tr23, data_[4], data_[5] - tr23
        );
    }

private:

    scalar data_[6];
};


KOKKOS_INLINE_FUNCTION
SymmTensor operator*(const scalar s, const SymmTensor& st) { return st * s; }

/** @brief Symmetric matrix-vector product S·v */
KOKKOS_INLINE_FUNCTION
Vec3 operator&(const SymmTensor& s, const Vec3& v)
{
    return Vec3(
        s(0, 0) * v[0] + s(0, 1) * v[1] + s(0, 2) * v[2],
        s(1, 0) * v[0] + s(1, 1) * v[1] + s(1, 2) * v[2],
        s(2, 0) * v[0] + s(2, 1) * v[1] + s(2, 2) * v[2]
    );
}

/** @brief Frobenius norm */
KOKKOS_INLINE_FUNCTION
scalar mag(const SymmTensor& s)
{
    // off-diagonal entries appear twice
    return sqrt(
        s(0, 0) * s(0, 0) + s(1, 1) * s(1, 1) + s(2, 2) * s(2, 2)
        + scalar(2) * (s(0, 1) * s(0, 1) + s(0, 2) * s(0, 2) + s(1, 2) * s(1, 2))
    );
}

/** @brief Frobenius norm squared */
KOKKOS_INLINE_FUNCTION
scalar magSqr(const SymmTensor& s)
{
    return s(0, 0) * s(0, 0) + s(1, 1) * s(1, 1) + s(2, 2) * s(2, 2)
         + scalar(2) * (s(0, 1) * s(0, 1) + s(0, 2) * s(0, 2) + s(1, 2) * s(1, 2));
}

/** @brief Symmetric part of a full tensor: ½(T + T^T) */
KOKKOS_INLINE_FUNCTION
SymmTensor symm(const Tensor& t)
{
    return SymmTensor(
        t(0, 0),
        0.5 * (t(0, 1) + t(1, 0)),
        0.5 * (t(0, 2) + t(2, 0)),
        t(1, 1),
        0.5 * (t(1, 2) + t(2, 1)),
        t(2, 2)
    );
}

/** @brief T + T^T (= 2*symm(T)) as a full tensor */
KOKKOS_INLINE_FUNCTION
Tensor twoSymm(const Tensor& t)
{
    return Tensor(
        scalar(2) * t(0, 0),
        t(0, 1) + t(1, 0),
        t(0, 2) + t(2, 0),
        t(1, 0) + t(0, 1),
        scalar(2) * t(1, 1),
        t(1, 2) + t(2, 1),
        t(2, 0) + t(0, 2),
        t(2, 1) + t(1, 2),
        scalar(2) * t(2, 2)
    );
}

/** @brief T + T^T - (2/3)*tr(T)*I — used for viscous stress */
KOKKOS_INLINE_FUNCTION
SymmTensor devTwoSymm(const Tensor& t)
{
    const scalar sph = (scalar(2) / scalar(3)) * (t(0, 0) + t(1, 1) + t(2, 2));
    return SymmTensor(
        scalar(2) * t(0, 0) - sph,
        t(0, 1) + t(1, 0),
        t(0, 2) + t(2, 0),
        scalar(2) * t(1, 1) - sph,
        t(1, 2) + t(2, 1),
        scalar(2) * t(2, 2) - sph
    );
}

std::ostream& operator<<(std::ostream& out, const SymmTensor& s);


template<>
KOKKOS_INLINE_FUNCTION SymmTensor zero<SymmTensor>()
{
    return SymmTensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

template<>
KOKKOS_INLINE_FUNCTION SymmTensor one<SymmTensor>()
{
    return SymmTensor(scalar(1));
}

template<>
KOKKOS_INLINE_FUNCTION SymmTensor inv<SymmTensor>(SymmTensor)
{
    return SymmTensor(scalar(1));
}


} // namespace NeoN
