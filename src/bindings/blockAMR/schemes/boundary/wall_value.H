// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// The wall sweep's MODES and SINKS (design §4.1, api §5.2) — B30a.
//
// This header names no method, no operator and no boundary condition, and it
// contains no arithmetic beyond a multiply-accumulate. It is the vocabulary a
// wall functor emits into; what a row is *for* is the sink's business and never
// the functor's.
//
// ---------------------------------------------------------------------------
// R1 — the affine split, held by the type system
// ---------------------------------------------------------------------------
// A wall row is `sum_k a_k phi_k + c`. `linear(i, j, k, a)` and `constant(c)`
// are two methods with two signatures, so a functor *cannot* route the BC datum
// through the linear part: the split is a type error rather than a review
// check. That is the whole enforcement, and it is why there are exactly two
// methods and why neither of them takes a flag.
//
// ---------------------------------------------------------------------------
// Why both sinks are device-legal PODs (plan §11, risk R1)
// ---------------------------------------------------------------------------
// A wall functor's `operator()` is `AMREX_GPU_HOST_DEVICE` and templated on the
// sink. Instantiating it with a sink whose methods are host-only makes nvcc
// emit a `__device__` instantiation that calls a `__host__` function — an error
// at the *instantiation*, whether or not it is ever launched. So `RecordSink`
// is a fixed-capacity POD with `AMREX_GPU_HOST_DEVICE` members and no
// allocation, not a `std::vector`, and `overflow` is a flag rather than a throw
// because a device-legal sink cannot throw. The binding checks the flag
// host-side and raises a sentence (design §10).
//
// `AssembleSink` — the implicit track's third sink (api §5.2) — is deliberately
// NOT built. It is named here so its absence is visible rather than forgotten;
// `WallMode::Assemble` is declared for the same reason, and the frame refuses
// it by name (S6: a missing capability raises naming itself).

#include <AMReX_Array4.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace ibm
{

//! How a wall row reaches its destination (design §6's composition rule).
//!
//! `Overwrite` for the first term of an equation, `Add` for every later one —
//! a per-call argument, because the driver flips it after the first term.
//! `Assemble` is the implicit track's seam and is declared, not implemented.
enum class WallMode : int
{
    Overwrite = 0,
    Add = 1,
    Assemble = 2
};

//! The explicit path: contract with `phi` immediately (api §5.2, verbatim).
//!
//! `constant_scale` is R2: `1.0` is the affine apply, `0.0` is the linear part
//! alone — the Krylov matvec — and it is one field rather than a second code
//! path.
struct ApplySink
{
    amrex::Array4<const amrex::Real> phi;
    int n;
    amrex::Real constant_scale;
    amrex::Real acc = 0.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
    linear(int i, int j, int k, amrex::Real a) noexcept
    {
        acc += a * phi(i, j, k, n);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void constant(amrex::Real c) noexcept
    {
        acc += constant_scale * c;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real value() const noexcept { return acc; }
};

//! The test path (api §10.6): a row, captured. Fixed capacity, POD, no
//! allocation — see the header note above for why it may not be a container.
//!
//! This is what gives the port back the row-level unit test v1 lost with its
//! row objects: the same `AMREX_GPU_HOST_DEVICE` functor that runs in the
//! kernel is called on the host for one cell, and the sink is read back.
struct RecordSink
{
    //! Sized from the widest row the ported functors emit, not from v1's
    //! `STRIDE`. `laplacian x ghostCell` (api §5.3) walks six arms; a wall arm
    //! emits one entry per trilinear donor, so a cell whose six face
    //! neighbours are all SOLID emits 6 * 8 = 48 linear entries. 64 is the
    //! next power of two above that, and `div`/`grad` (B33/B34) are the same
    //! six arms with a different weight. A row that does overflow sets the
    //! flag; it is never silently truncated.
    static constexpr int capacity = 64;

    struct Entry
    {
        int i, j, k;
        amrex::Real a;
    };

    Entry entries[capacity];
    int count = 0;
    amrex::Real c = 0.0;
    bool overflow = false;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
    linear(int i, int j, int k, amrex::Real a) noexcept
    {
        if (count == capacity)
        {
            overflow = true;
            return;
        }
        entries[count] = Entry {i, j, k, a};
        ++count;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void constant(amrex::Real v) noexcept { c += v; }
};

} // namespace ibm
