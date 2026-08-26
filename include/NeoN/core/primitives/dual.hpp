// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp> // IWYU pragma: keep

#include "NeoN/core/primitives/traits.hpp"

namespace NeoN
{

/**
 * @class Dual
 * @brief Forward-mode automatic differentiation primitive.
 *
 * Carries a primal value together with a compile-time fixed number of
 * directional derivatives. Because NDeriv is a template parameter the type has
 * no dynamic state: it is trivially copyable, contains no pointers and requires
 * no allocation. This is what makes it usable inside Kokkos device kernels,
 * unlike tape-based AD types which must append to a runtime-growing trace.
 *
 * Memory layout is Array-of-Struct. For NDeriv <= ~4 this is generally the
 * faster choice on device since the derivatives of a cell are consumed together
 * with its value. For larger NDeriv a Struct-of-Array layout with a hidden
 * derivative dimension (cf. Sacado's Kokkos View support) should be measured
 * before committing - see doc/ad.md.
 *
 * @tparam ValueType  underlying arithmetic type, typically NeoN::scalar
 * @tparam NDeriv     number of design variables carried simultaneously
 *
 * @ingroup Primitives
 */
template<typename ValueType, int NDeriv>
class Dual
{
public:

    using PrimalType = ValueType;
    static constexpr int nDeriv = NDeriv;

    KOKKOS_INLINE_FUNCTION
    Dual() : v_(ValueType(0))
    {
        for (int i = 0; i < NDeriv; ++i) d_[i] = ValueType(0);
    }

    /** @brief Passive constructor: a constant has zero derivative. */
    KOKKOS_INLINE_FUNCTION
    Dual(ValueType v) : v_(v)
    {
        for (int i = 0; i < NDeriv; ++i) d_[i] = ValueType(0);
    }

    /**
     * @brief Active constructor: seeds slot with unit derivative.
     * @param v     primal value
     * @param slot  index of the design variable this value *is*
     */
    KOKKOS_INLINE_FUNCTION
    Dual(ValueType v, int slot) : v_(v)
    {
        for (int i = 0; i < NDeriv; ++i) d_[i] = ValueType(0);
        if (slot >= 0 && slot < NDeriv) d_[slot] = ValueType(1);
    }

    KOKKOS_INLINE_FUNCTION ValueType value() const { return v_; }
    KOKKOS_INLINE_FUNCTION ValueType& value() { return v_; }

    KOKKOS_INLINE_FUNCTION ValueType deriv(int i) const { return d_[i]; }
    KOKKOS_INLINE_FUNCTION ValueType& deriv(int i) { return d_[i]; }

    // --- compound assignment ------------------------------------------------

    KOKKOS_INLINE_FUNCTION Dual& operator+=(const Dual& r)
    {
        v_ += r.v_;
        for (int i = 0; i < NDeriv; ++i) d_[i] += r.d_[i];
        return *this;
    }

    KOKKOS_INLINE_FUNCTION Dual& operator-=(const Dual& r)
    {
        v_ -= r.v_;
        for (int i = 0; i < NDeriv; ++i) d_[i] -= r.d_[i];
        return *this;
    }

    KOKKOS_INLINE_FUNCTION Dual& operator*=(const Dual& r)
    {
        for (int i = 0; i < NDeriv; ++i) d_[i] = d_[i] * r.v_ + v_ * r.d_[i];
        v_ *= r.v_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION Dual& operator/=(const Dual& r)
    {
        const ValueType invR = ValueType(1) / r.v_;
        for (int i = 0; i < NDeriv; ++i) d_[i] = (d_[i] - v_ * invR * r.d_[i]) * invR;
        v_ *= invR;
        return *this;
    }

    // --- binary arithmetic --------------------------------------------------
    // Hidden friends rather than namespace-scope templates: a template taking
    // Dual<T, N> on both sides cannot deduce T from a bare ValueType operand,
    // so the ubiquitous `scalar * dual` would be a hard compile error. As
    // friends the parameters are the concrete specialisation and a scalar
    // converts through the passive constructor, as intended.

    friend KOKKOS_INLINE_FUNCTION Dual operator+(Dual a, const Dual& b) { return a += b; }

    friend KOKKOS_INLINE_FUNCTION Dual operator-(Dual a, const Dual& b) { return a -= b; }

    friend KOKKOS_INLINE_FUNCTION Dual operator*(Dual a, const Dual& b) { return a *= b; }

    friend KOKKOS_INLINE_FUNCTION Dual operator/(Dual a, const Dual& b) { return a /= b; }

    friend KOKKOS_INLINE_FUNCTION Dual operator-(const Dual& a)
    {
        Dual r(-a.v_);
        for (int i = 0; i < NDeriv; ++i) r.d_[i] = -a.d_[i];
        return r;
    }

    // --- comparison ---------------------------------------------------------
    // Comparisons act on the primal only. This is the standard convention and
    // it is also where forward-mode AD silently loses information: a branch
    // taken on a comparison is not differentiated, so limiters and upwind
    // switches yield the derivative of whichever branch was selected. See the
    // differentiability audit in doc/ad.md.

    friend KOKKOS_INLINE_FUNCTION bool operator<(const Dual& a, const Dual& b)
    {
        return a.v_ < b.v_;
    }

    friend KOKKOS_INLINE_FUNCTION bool operator>(const Dual& a, const Dual& b)
    {
        return a.v_ > b.v_;
    }

    friend KOKKOS_INLINE_FUNCTION bool operator<=(const Dual& a, const Dual& b)
    {
        return a.v_ <= b.v_;
    }

    friend KOKKOS_INLINE_FUNCTION bool operator>=(const Dual& a, const Dual& b)
    {
        return a.v_ >= b.v_;
    }

    friend KOKKOS_INLINE_FUNCTION bool operator==(const Dual& a, const Dual& b)
    {
        return a.v_ == b.v_;
    }

    friend KOKKOS_INLINE_FUNCTION bool operator!=(const Dual& a, const Dual& b)
    {
        return a.v_ != b.v_;
    }

private:

    ValueType v_;
    ValueType d_[NDeriv];
};

// --- elementary functions ---------------------------------------------------

template<typename T, int N>
KOKKOS_INLINE_FUNCTION Dual<T, N> sqrt(const Dual<T, N>& a)
{
    using Kokkos::sqrt;
    const T s = sqrt(a.value());
    Dual<T, N> r(s);
    const T c = T(0.5) / s;
    for (int i = 0; i < N; ++i) r.deriv(i) = c * a.deriv(i);
    return r;
}

template<typename T, int N>
KOKKOS_INLINE_FUNCTION Dual<T, N> exp(const Dual<T, N>& a)
{
    using Kokkos::exp;
    const T e = exp(a.value());
    Dual<T, N> r(e);
    for (int i = 0; i < N; ++i) r.deriv(i) = e * a.deriv(i);
    return r;
}

template<typename T, int N>
KOKKOS_INLINE_FUNCTION Dual<T, N> log(const Dual<T, N>& a)
{
    using Kokkos::log;
    Dual<T, N> r(log(a.value()));
    const T c = T(1) / a.value();
    for (int i = 0; i < N; ++i) r.deriv(i) = c * a.deriv(i);
    return r;
}

/** @brief Magnitude. Non-differentiable at zero; derivative there is set to 0. */
template<typename T, int N>
KOKKOS_INLINE_FUNCTION Dual<T, N> mag(const Dual<T, N>& a)
{
    const T s = (a.value() > T(0)) ? T(1) : ((a.value() < T(0)) ? T(-1) : T(0));
    Dual<T, N> r(s * a.value());
    for (int i = 0; i < N; ++i) r.deriv(i) = s * a.deriv(i);
    return r;
}

// --- NeoN traits ------------------------------------------------------------

template<typename T, int N>
KOKKOS_INLINE_FUNCTION Dual<T, N> oneDual()
{
    return Dual<T, N>(T(1));
}

template<typename T, int N>
KOKKOS_INLINE_FUNCTION Dual<T, N> zeroDual()
{
    return Dual<T, N>(T(0));
}

template<typename T, int N>
KOKKOS_INLINE_FUNCTION Dual<T, N> inv(Dual<T, N> in)
{
    return Dual<T, N>(T(1)) /= in;
}

} // namespace NeoN

/**
 * @brief Register NeoN::one/zero traits for a concrete Dual instantiation.
 *
 * NeoN's traits are *function* templates (template<typename T> T one();), and
 * C++ does not permit partial specialization of function templates. Dual has
 * two template parameters, so each instantiation must be registered explicitly.
 *
 * This macro is a stopgap. The correct upstream fix is to convert
 * core/primitives/traits.hpp to struct traits:
 *
 *     template<typename T> struct Traits { static KOKKOS_INLINE_FUNCTION T one(); };
 *
 * which partially specializes cleanly and removes the need to enumerate N.
 * That refactor touches every existing primitive, so it is deliberately kept
 * out of this MWE.
 */
#define NeoN_DUAL_REGISTER_TRAITS(TYPE, N)                                                         \
    namespace NeoN                                                                                 \
    {                                                                                              \
    template<>                                                                                     \
    KOKKOS_INLINE_FUNCTION Dual<TYPE, N> one<Dual<TYPE, N>>()                                      \
    {                                                                                              \
        return oneDual<TYPE, N>();                                                                 \
    }                                                                                              \
    template<>                                                                                     \
    KOKKOS_INLINE_FUNCTION Dual<TYPE, N> zero<Dual<TYPE, N>>()                                     \
    {                                                                                              \
        return zeroDual<TYPE, N>();                                                                \
    }                                                                                              \
    }
