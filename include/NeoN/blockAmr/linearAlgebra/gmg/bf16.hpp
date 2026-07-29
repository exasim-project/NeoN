// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

#include <cstdint>

namespace blockamr::la
{

// bfloat16 (FP32's 8-bit exponent, ~3 decimal digits) as a STORAGE type for the GMG
// level hierarchy. STORAGE ONLY is a correctness requirement: every operation below
// converts to float, computes there and rounds once on the way back, and the kernels
// compute in GmgComputeT<T> — accumulated in bf16, the V-cycle's residual cancels to
// exactly 0.0. Measured and REJECTED for the FIELDS; kept because bf16 COEFFICIENTS
// under fp32 fields do win (measurements:
// report/blockamr-precision-measurements.md in the NeoFOAM repo).
struct Bf16
{
    std::uint16_t bits;

    // Trivially default constructible on purpose: BaseFab's placementNew is then a
    // no-op, so a bf16 level costs the same allocation path as a float one.
    Bf16() noexcept = default;

    // The ONLY converting constructor: double, int and friends reach it through the
    // standard conversion to float, and a second one taking double would make
    // Bf16(0) ambiguous.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Bf16(float f) noexcept : bits(fromFloat(f)) {}

    // Implicit, so every expression a kernel writes over a bf16 level promotes to
    // float and computes there. Not ambiguous against the double built-ins: Bf16 ->
    // float is an exact match after the user-defined conversion, Bf16 -> double adds
    // a floating-point promotion on top of it.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE operator float() const noexcept
    {
        return toFloat(bits);
    }

    // `x += y` on a class type never finds a built-in candidate, and the prolongation
    // is written that way. Load, add in float, round once.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Bf16& operator+=(float rhs) noexcept
    {
        bits = fromFloat(toFloat(bits) + rhs);
        return *this;
    }

    // Round to nearest, ties to even: truncation is two integer ops cheaper and twice
    // the error, and these kernels wait on memory, not the ALU. A NaN is mapped to a
    // quiet NaN rather than left to round its payload into an infinity.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static std::uint16_t fromFloat(float f) noexcept
    {
        Bits b;
        b.f = f;
        if ((b.u & 0x7F800000u) == 0x7F800000u && (b.u & 0x007FFFFFu) != 0u)
        {
            return 0x7FC0u;
        }
        const std::uint32_t rounding = 0x7FFFu + ((b.u >> 16) & 1u);
        return static_cast<std::uint16_t>((b.u + rounding) >> 16);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static float toFloat(std::uint16_t h) noexcept
    {
        Bits b;
        b.u = static_cast<std::uint32_t>(h) << 16;
        return b.f;
    }

private:

    // Type punning through a union: not strictly conforming C++, but the form GCC,
    // Clang and nvcc all document as supported, and std::bit_cast is not usable in
    // device code across our toolchains.
    union Bits
    {
        float f;
        std::uint32_t u;
    };
};

static_assert(sizeof(Bf16) == 2, "Bf16 must be exactly two bytes");

// The type the KERNELS compute in, given the type a LEVEL is STORED in. Identity for
// float and double, so those paths generate exactly the code they did before this
// type existed (their bit-exactness tests gate that), and float for bf16, which keeps
// the level diagonal exact (rounding argument:
// report/blockamr-precision-measurements.md in the NeoFOAM repo).
template<class T>
struct GmgCompute
{
    using type = T;
};

template<>
struct GmgCompute<Bf16>
{
    using type = float;
};

template<class T>
using GmgComputeT = typename GmgCompute<T>::type;

} // namespace blockamr::la
