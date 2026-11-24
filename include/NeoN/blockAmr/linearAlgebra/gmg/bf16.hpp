// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

#include <cstdint>

namespace blockamr::la
{

// bfloat16 as a STORAGE type for the GMG level hierarchy: every operation below converts
// to float and rounds once back. Rejected for the FIELDS, kept for the COEFFICIENTS —
// report/blockamr-precision-measurements.md (NeoFOAM repo).
struct Bf16
{
    std::uint16_t bits;

    // Trivially default constructible: BaseFab's placementNew is then a no-op.
    Bf16() noexcept = default;

    // The only converting constructor: a double overload would make Bf16(0) ambiguous.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Bf16(float f) noexcept : bits(fromFloat(f)) {}

    // Implicit, so kernel expressions over a bf16 level promote to float and compute there.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE operator float() const noexcept
    {
        return toFloat(bits);
    }

    // A class type finds no built-in +=, and the prolongation is written that way.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Bf16& operator+=(float rhs) noexcept
    {
        bits = fromFloat(toFloat(bits) + rhs);
        return *this;
    }

    // Round to nearest, ties to even; a NaN maps to a quiet NaN rather than an infinity.
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

    // Union punning: what GCC, Clang and nvcc all support; std::bit_cast is not
    // device-usable across our toolchains.
    union Bits
    {
        float f;
        std::uint32_t u;
    };
};

static_assert(sizeof(Bf16) == 2, "Bf16 must be exactly two bytes");

// The type the KERNELS compute in, given a level's STORAGE type: identity except float
// for Bf16, which keeps the level diagonal exact (report/blockamr-precision-measurements.md).
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
