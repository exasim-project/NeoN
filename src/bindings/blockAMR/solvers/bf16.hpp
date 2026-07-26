// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

#include <cstdint>

namespace blockamr::solvers
{

// ---------------------------------------------------------------------------
// bfloat16 as a STORAGE type for the GMG level hierarchy: 1 sign + 8 exponent +
// 7 mantissa bits, i.e. FP32's exponent range in half of FP32's bytes. The
// V-cycle is bandwidth-bound once the launch cost is gone, so what a level's
// value type buys is bytes moved, not flops.
//
// Why bf16 and not IEEE fp16, for THIS operator: the face coefficient of the
// Laplacian is -beta/dx^2, which on a unit cube is -65536 at 256^3 and -262144
// at 512^3. IEEE half tops out at 65504, so the coefficients themselves would
// overflow to infinity before any arithmetic happened. bf16 keeps FP32's
// exponent, so dynamic range is not the constraint here -- only the ~3
// significant decimal digits are.
//
// STORAGE ONLY, and that is a correctness requirement rather than a nicety.
// Every operation below converts to float, computes in float and rounds once on
// the way back; the kernels declare their locals as GmgComputeT<T> (= float
// here) rather than T. What forces that is the residual, which the V-cycle forms
// as a difference of two quantities vastly larger than itself:
//
//     r = b - (diag * psi + off),   diag = alpha - sum(a_face) = 1 + 6/dx^2
//
// At 256^3 with psi ~ 0.7 that is diag*psi = 275252 against off = -275212, whose
// sum is 40. Round either intermediate to bf16 -- spacing 2048 up there -- and
// both land on the SAME bf16 value, so the difference comes out exactly 0.0:
// 100% of the residual gone, at every grid size (the same experiment at 16^3
// loses all of 0.85). Kept in float, the subtraction is exact given its inputs
// and only the ~0.4% the stored values carry survives.
//
// The diagonal itself is a red herring by comparison: bf16 does round 393217 to
// 393216 and lose alpha, but alpha's share of that diagonal is 2.5e-6, two
// orders below bf16's own representation error. Storing the coefficients in
// bf16 is fine; accumulating in it is not.
//
// WHAT IT MEASURED: A NEGATIVE RESULT, KEPT BECAUSE IT IS ONE
//
// The bytes arrive. The 256^3/512-box V-cycle drops from 11.96 ms at fp32 to
// 8.82 ms, 1.36x. It is not enough, at any size measured.
//
// The reason is amplification, and it is specific to what a V-cycle does with a
// stored solution. Holding psi at ~0.4% puts a per-cell perturbation d into it,
// and the quantity the cycle restricts to the coarse grid is r = b - A(psi + d),
// so d arrives there multiplied by ||A||. For this operator ||A|| ~ 6/dx^2 =
// 6n^2: the noise floor of the restricted residual grows as n^2 while the
// residual itself does not. The coarse grid is then correcting noise.
//
// One V-cycle's residual reduction against the same fp64 cycle, and the CG
// iterations that costs (256^3/512 boxes, norm=linf, l0 agglomerated):
//
//     grid    V-cycle weaker    fp32 iters    bf16 iters    solve vs fp32
//     16^3         1.05x            --            --             --
//     64^3         1.26x            11            25           1.9x slower
//     128^3        1.87x            11            53           3.6x slower
//     256^3        3.23x            12           273          17.4x slower
//
// The answers stay correct throughout -- the operator and the residual CG stops
// on are fp64 whatever the hierarchy is stored in -- so this is purely a worse
// preconditioner, never a wrong one. There is no crossover: 1.36x off the
// V-cycle cannot pay for doubling the iteration count, which already happens at
// 64^3.
//
// It is kept, wired and tested rather than deleted because a measured "no" is
// worth more than an untested "probably not", and because the parts generalise:
// GmgComputeT is what any reduced-precision level will need, and the precision
// axis is now first-class in the bench.
//
// THE REFINEMENT THE NUMBERS POINTED AT, AND WHAT IT MEASURED
//
// Storing the COEFFICIENTS in bf16 while psi and rhs stay fp32 -- a coefficient
// error is a 0.4% perturbation of the operator, which a preconditioner absorbs
// without amplification, and it is 4 of the 6 arrays a shared-coefficient colour
// sweep streams. That is `gmg_coeff_precision`, and it works. 256^3, one box,
// level-0 agglomerated, one V-cycle from z0 = 0:
//
//     fields/coeffs   ms/cycle   r1/r0 (smooth b)   CG iters   solve
//     fp32 / fp32       12.52         0.70185           9      213 ms
//     fp32 / bf16       10.60         0.70147           9      195 ms
//     bf16 / bf16        9.37        97.7 (!)          --        --
//
// The middle column is the whole argument. Narrowing the COEFFICIENTS leaves the
// cycle's residual reduction where it was -- 0.70147 against 0.70185, a 0.05%
// difference, and in the favourable direction -- while narrowing the FIELDS as
// well turns a contraction into a 98x AMPLIFICATION at this size. Same storage
// type, same kernels, same 3 decimal digits: the difference is only which array
// carries them, and whether ||A|| ~ 6/dx^2 multiplies the error before the coarse
// grid sees it.
//
// One negative result inside the positive one: fp64 FIELDS with bf16 coefficients
// is 23.82 -> 26.54 ms, i.e. 1.11x SLOWER, at a cycle strength identical to five
// digits. Narrowing is only worth it once the fields are narrow too. The mechanism
// was not isolated (no ncu run); the arithmetic is the visible difference --
// GmgComputeT<double> is double, so every coefficient there is unpacked to float
// and then widened again, where the fp32 path stops at the float bf16 natively
// converts to.
// ---------------------------------------------------------------------------
struct Bf16
{
    std::uint16_t bits;

    // Trivially default constructible on purpose: BaseFab's placementNew is then
    // a no-op (AMReX_BaseFab.H:102), so a bf16 level costs the same allocation
    // path as a float one.
    Bf16() noexcept = default;

    // The ONLY converting constructor. double, int and friends reach it through
    // the standard conversion to float that may precede a user-defined one;
    // adding a second constructor taking double would make Bf16(0) ambiguous.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Bf16(float f) noexcept : bits(fromFloat(f)) {}

    // Implicit, so that every expression a kernel writes over a bf16 level --
    // `aE * psi(i + 1, j, k)`, `b(i, j, k) - off` -- promotes to float and
    // computes there. Overload resolution is not ambiguous against the
    // double built-ins: Bf16 -> float is an exact match after the user-defined
    // conversion, Bf16 -> double adds a floating-point promotion on top of it.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE operator float() const noexcept
    {
        return toFloat(bits);
    }

    // `x += y` on a class type never finds a built-in candidate, and the
    // prolongation is written that way. Load, add in float, round once: the
    // storage-only contract in one line.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Bf16& operator+=(float rhs) noexcept
    {
        bits = fromFloat(toFloat(bits) + rhs);
        return *this;
    }

    // Round to nearest, ties to even. Truncation would be two integer ops
    // cheaper and twice the error; these kernels wait on memory, not on the
    // ALU, so the error is the side that matters. A NaN is mapped to a quiet
    // NaN rather than left to round its payload into an infinity.
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

    // Type punning through a union: not strictly conforming C++, but the form
    // GCC, Clang and nvcc all document as supported, and the one that needs no
    // <bit> (std::bit_cast is not usable in device code across our toolchains).
    union Bits
    {
        float f;
        std::uint32_t u;
    };
};

static_assert(sizeof(Bf16) == 2, "Bf16 must be exactly two bytes");

// The type the KERNELS compute in, given the type a LEVEL is STORED in.
//
// Identity for float and double, so those two paths generate exactly the code
// they did before this type existed -- their bit-exactness tests are the gate on
// that -- and float for bf16, which is what keeps the diagonal above exact.
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

} // namespace blockamr::solvers
