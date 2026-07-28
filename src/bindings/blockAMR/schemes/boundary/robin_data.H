// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// The BC transport (Q4, design §8, api §5.3) — B30a.
//
// How `alpha`, `beta` and `gamma(t)` reach a wall kernel, and NOTHING about
// what the kernel does with them. There is no closure here, no distance, no
// branch on the BC type, no `1/d`: the wall formula is `robin.H`'s
// `closure(alpha, beta, gamma, d)` (B30b), which is the row under revision
// (review.md §4 Q28/Q30, condition C1) and which no file in B30a includes.
//
// ---------------------------------------------------------------------------
// gamma(t) IS A COMPILED EXPRESSION, NOT A CALLBACK (Q4, Q29e)
// ---------------------------------------------------------------------------
// design §8 asks for "a gamma read inside the kernel", and Q4 rules a Python
// callable out of scope for v2. `GammaExpr` is that read: a 40-byte POD per
// (patch, component), uploaded once per evaluate and evaluated by one
// `AMREX_GPU_HOST_DEVICE operator()(t)`.
//
// The basis is `a0 + ac*cos(omega t) + as*sin(omega t)` rather than an
// amplitude/phase pair because it is a BITWISE respelling of the two users the
// tree actually has: A4 (Stokes' second problem) is `U0 cos(omega t)`, i.e.
// `(a0 = 0, ac = U0, as = 0)`, and A6 (Womersley) is `-(G/omega) sin(omega t)`,
// i.e. `(a0 = 0, ac = 0, as = -G/omega)`. An amplitude/phase form would spell
// A6 as `cos(omega t - pi/2)`, which is not bitwise `sin(omega t)` — it would
// move A4/A6's fitted numbers in the last bits in the very session that has to
// show the port did not move them. The cos/sin basis is also the one A4/A6's
// own `_fit_harmonic` uses to measure them.
//
// `Constant` is an explicit TAG rather than a harmonic with `ac = as = 0`
// because verification §2's probes assert *exactly* zero on constant data
// (rungs 1-5, `laplacian(1, T)` with `T = C`). The tag keeps a transcendental
// off that path entirely instead of relying on `0.0 * cos(x)` being exactly
// `+-0.0` in every build; it costs one int and one branch that is uniform
// across a patch.
//
// What it deliberately does not carry: a SPATIAL datum. Q25's OP-1 shipped a
// callable of `t` only on v1 and left A3's rotating wall out of scope; when
// A3 lands, the spatial form is a new `Form` tag plus an overload, and the
// wall foot point it needs is already in `IbmGeometryView`, which the functor
// already holds — no change to this layout, to the entry point, or to
// `wall_apply.H`.

#include <AMReX_Extension.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ibm
{

//! `gamma(t)` for one (patch, component), as a compiled per-patch expression.
struct GammaExpr
{
    enum Form : int
    {
        Constant = 0,
        Harmonic = 1
    };

    int form = Constant;
    amrex::Real a0 = 0.0;    //!< Constant: the value. Harmonic: the mean.
    amrex::Real ac = 0.0;    //!< cos amplitude
    amrex::Real as = 0.0;    //!< sin amplitude
    amrex::Real omega = 0.0; //!< angular frequency

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(amrex::Real t) const noexcept
    {
        if (form == Constant) return a0;
        return a0 + ac * std::cos(omega * t) + as * std::sin(omega * t);
    }
};

//! The device-copyable POD a wall functor captures by value, read-only by
//! construction like `IbmGeometryView` and `GhostCellView`.
//!
//! `alpha` and `beta` are one scalar per patch — v1's `SurfaceBC.robin()`
//! returns scalars for all three BC types and only gamma is ever
//! per-component. `gamma` is row-major `(patch, component)`.
//!
//! The patch index is `IbmGeometryView::patch`, i.e. the position of the body
//! in `sorted(mesh.bodies)` (`ibm/classify.py::_patches`), which is the same
//! enumeration v1's band tables index. A `RobinData` built on a different
//! enumeration applies the wrong BC to the wrong body silently, so the builder
//! owns that invariant; see the note on `npatch` in `requireRobinComponents`.
struct RobinView
{
    const amrex::Real* alpha;
    const amrex::Real* beta;
    const GammaExpr* gamma;
    int npatch;
    int ncomp;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    gammaAt(int patch, int n, amrex::Real t) const noexcept
    {
        return gamma[patch * ncomp + n](t);
    }
};

//! Owns the `(alpha, beta, gamma)` tables; built once per (field, evaluate) and
//! passed by const reference.
//!
//! The storage is MANAGED, not device-only, and that is load-bearing rather
//! than convenient: `RobinView` is captured by a functor that api §10.6 and
//! tasks.md §3's verify column require to be callable **on the host for one
//! cell**. A device-only allocation would make `gammaAt` a host segfault and
//! would force a second, differently-typed host view — at which point the
//! host-side row test would no longer be exercising the view the kernel uses.
//! The tables are `npatch * ncomp` entries of 40 bytes; there is no bandwidth
//! argument on either side.
class RobinData
{
public:

    //! `alpha[npatch]`, `beta[npatch]`, `form[npatch][ncomp]`,
    //! `param[npatch][ncomp][4]` with `param` = (a0, ac, as, omega).
    //! Shapes are the caller's business (the binding checks them against the
    //! numpy arrays it was handed); what is checked here is what survives the
    //! loss of the shapes.
    RobinData(
        const amrex::Real* alpha,
        const amrex::Real* beta,
        const int* form,
        const amrex::Real* param,
        int npatch,
        int ncomp
    )
        : npatch_(npatch), ncomp_(ncomp)
    {
        if (npatch < 1 || ncomp < 1)
            throw std::runtime_error(
                "RobinData: a Robin table needs at least one patch and one component, but was "
                "given npatch = "
                + std::to_string(npatch) + " and ncomp = " + std::to_string(ncomp)
            );

        std::vector<amrex::Real> ha(alpha, alpha + npatch);
        std::vector<amrex::Real> hb(beta, beta + npatch);
        std::vector<GammaExpr> hg(static_cast<std::size_t>(npatch) * ncomp);

        for (int p = 0; p < npatch; ++p)
            for (int n = 0; n < ncomp; ++n)
            {
                const std::size_t r = static_cast<std::size_t>(p) * ncomp + n;
                const int f = form[r];
                if (f != GammaExpr::Constant && f != GammaExpr::Harmonic)
                    throw std::runtime_error(
                        "RobinData: gamma(t) is a compiled expression with two forms, "
                        "Constant (0) and Harmonic (1), but patch "
                        + std::to_string(p) + " component " + std::to_string(n) + " asks for form "
                        + std::to_string(f)
                        + "; a datum that is neither is not expressible device-side (Q4)"
                    );
                hg[r] = GammaExpr {
                    f, param[4 * r + 0], param[4 * r + 1], param[4 * r + 2], param[4 * r + 3]
                };
            }

        alpha_.resize(npatch);
        beta_.resize(npatch);
        gamma_.resize(hg.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, ha.begin(), ha.end(), alpha_.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, hb.begin(), hb.end(), beta_.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, hg.begin(), hg.end(), gamma_.begin());
        amrex::Gpu::streamSynchronize();
    }

    RobinView view() const noexcept
    {
        return RobinView {alpha_.data(), beta_.data(), gamma_.data(), npatch_, ncomp_};
    }

    int npatch() const noexcept { return npatch_; }
    int ncomp() const noexcept { return ncomp_; }

private:

    amrex::Gpu::ManagedVector<amrex::Real> alpha_;
    amrex::Gpu::ManagedVector<amrex::Real> beta_;
    amrex::Gpu::ManagedVector<GammaExpr> gamma_;
    int npatch_;
    int ncomp_;
};

//! The one guard a pair's binding must make for itself, because it is the only
//! guard that needs to know what a robin is (`wall_apply.H` deliberately does
//! not include this header — that is the formula-independence claim, stated
//! mechanically).
//!
//! A `robin` with fewer components than the field is a silent out-of-bounds
//! read of `gamma`, not a wrong number, which is exactly the class of failure
//! api §9 exists to turn into a sentence.
//!
//! `npatch` is NOT cross-checked against the geometry here: the packed
//! geometry fab publishes no patch count (its `patch` component is a per-cell
//! id), so the "every patch id in the geometry is < robin.npatch" invariant is
//! the Python builder's until something carries the count. One line here, the
//! day it does.
inline void requireRobinComponents(const char* fn, int robin_ncomp, int ncomp)
{
    if (robin_ncomp == ncomp) return;
    throw std::runtime_error(
        std::string(fn)
        + ": the Robin table is indexed (patch, component), so it must carry one "
          "gamma per field component — the field has "
        + std::to_string(ncomp) + " but the table has " + std::to_string(robin_ncomp)
        + "; rebuild the Robin table for this field"
    );
}

} // namespace ibm
