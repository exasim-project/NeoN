// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp> // IWYU pragma: keep

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/primitives/traits.hpp"


namespace NeoN
{


/**
 * @class Tensor
 * @brief A class for the representation of a 3x3 tensor
 * @ingroup primitives
 *
 * Row-major storage: component (i,j) is at data_[3*i+j].
 */
class Tensor
{
public:

    KOKKOS_INLINE_FUNCTION
    Tensor()
    {
        for (int k = 0; k < 9; ++k)
        {
            data_[k] = 0.0;
        }
    }

    /** @brief Construct scalar-times-identity tensor */
    KOKKOS_INLINE_FUNCTION
    explicit Tensor(const scalar diag)
    {
        for (int k = 0; k < 9; ++k)
        {
            data_[k] = 0.0;
        }
        data_[0] = diag;
        data_[4] = diag;
        data_[8] = diag;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor(
        scalar t00,
        scalar t01,
        scalar t02,
        scalar t10,
        scalar t11,
        scalar t12,
        scalar t20,
        scalar t21,
        scalar t22
    )
    {
        data_[0] = t00;
        data_[1] = t01;
        data_[2] = t02;
        data_[3] = t10;
        data_[4] = t11;
        data_[5] = t12;
        data_[6] = t20;
        data_[7] = t21;
        data_[8] = t22;
    }

    KOKKOS_INLINE_FUNCTION scalar* data() { return data_; }
    KOKKOS_INLINE_FUNCTION const scalar* data() const { return data_; }

    constexpr size_t size() const { return 9; }

    KOKKOS_INLINE_FUNCTION
    scalar& operator()(const size_t i, const size_t j) { return data_[3 * i + j]; }

    KOKKOS_INLINE_FUNCTION
    scalar operator()(const size_t i, const size_t j) const { return data_[3 * i + j]; }

    KOKKOS_INLINE_FUNCTION
    bool operator==(const Tensor& rhs) const
    {
        for (int k = 0; k < 9; ++k)
        {
            if (data_[k] != rhs.data_[k]) return false;
        }
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor operator+(const Tensor& rhs) const
    {
        Tensor res;
        for (int k = 0; k < 9; ++k)
        {
            res.data_[k] = data_[k] + rhs.data_[k];
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor& operator+=(const Tensor& rhs)
    {
        for (int k = 0; k < 9; ++k)
        {
            data_[k] += rhs.data_[k];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor operator-(const Tensor& rhs) const
    {
        Tensor res;
        for (int k = 0; k < 9; ++k)
        {
            res.data_[k] = data_[k] - rhs.data_[k];
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor& operator-=(const Tensor& rhs)
    {
        for (int k = 0; k < 9; ++k)
        {
            data_[k] -= rhs.data_[k];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor operator*(const scalar s) const
    {
        Tensor res;
        for (int k = 0; k < 9; ++k)
        {
            res.data_[k] = data_[k] * s;
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    Tensor& operator*=(const scalar s)
    {
        for (int k = 0; k < 9; ++k)
        {
            data_[k] *= s;
        }
        return *this;
    }

    /** @brief Trace: sum of diagonal components */
    KOKKOS_INLINE_FUNCTION
    scalar trace() const { return data_[0] + data_[4] + data_[8]; }

    /** @brief Transpose */
    KOKKOS_INLINE_FUNCTION
    Tensor T() const // NOLINT
    {
        return Tensor(
            data_[0], data_[3], data_[6], data_[1], data_[4], data_[7], data_[2], data_[5], data_[8]
        );
    }

private:

    scalar data_[9];
};


KOKKOS_INLINE_FUNCTION
Tensor operator*(const scalar s, const Tensor& t) { return t * s; }

/** @brief Matrix-vector product T·v */
KOKKOS_INLINE_FUNCTION
Vec3 operator&(const Tensor& t, const Vec3& v)
{
    return Vec3(
        t(0, 0) * v[0] + t(0, 1) * v[1] + t(0, 2) * v[2],
        t(1, 0) * v[0] + t(1, 1) * v[1] + t(1, 2) * v[2],
        t(2, 0) * v[0] + t(2, 1) * v[1] + t(2, 2) * v[2]
    );
}

/** @brief Row-vector times matrix v·T */
KOKKOS_INLINE_FUNCTION
Vec3 operator&(const Vec3& v, const Tensor& t)
{
    return Vec3(
        v[0] * t(0, 0) + v[1] * t(1, 0) + v[2] * t(2, 0),
        v[0] * t(0, 1) + v[1] * t(1, 1) + v[2] * t(2, 1),
        v[0] * t(0, 2) + v[1] * t(1, 2) + v[2] * t(2, 2)
    );
}

/** @brief Frobenius norm */
KOKKOS_INLINE_FUNCTION
scalar mag(const Tensor& t)
{
    scalar s = 0;
    for (int k = 0; k < 9; ++k)
    {
        s += t.data()[k] * t.data()[k];
    }
    return sqrt(s);
}

/** @brief Frobenius norm squared */
KOKKOS_INLINE_FUNCTION
scalar magSqr(const Tensor& t)
{
    scalar s = 0;
    const scalar* d = t.data();
    for (int k = 0; k < 9; ++k)
    {
        s += d[k] * d[k];
    }
    return s;
}

std::ostream& operator<<(std::ostream& out, const Tensor& t);


template<>
KOKKOS_INLINE_FUNCTION Tensor zero<Tensor>()
{
    return Tensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

template<>
KOKKOS_INLINE_FUNCTION Tensor one<Tensor>()
{
    return Tensor(scalar(1));
}

template<>
KOKKOS_INLINE_FUNCTION Tensor inv<Tensor>(Tensor)
{
    // Placeholder — matrix inverse not needed for field arithmetic
    return Tensor(scalar(1));
}


} // namespace NeoN
