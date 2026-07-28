// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The closure's translation unit (B30b) — the FORMULA half of B30.
//
// `robin.H` is a header; something has to instantiate it, on the host and on
// the device, and something has to hand its numbers back to a numpy peer. That
// is all this file is. It is TEST-ONLY: two underscore-private hooks (api §4),
// no `WALL_SCHEMES` entry, no `wall_<operator>_<method>` name, nothing
// reachable from `src/blockamr/`, and no pair. The real
// `laplacian x ghostCell` pair is B32.
//
// It is a separate TU from `wall_frame.cpp` on purpose (review.md §4 Q43(a)):
// `wall_frame.cpp` carries a shipped, reviewed floating-point decision in its
// own header — no per-file flags, because nothing in it is pinned bitwise — and
// its row count is itself evidence that B30b did not move the frame. This file
// needs the opposite posture, so it gets its own.
//
// ---------------------------------------------------------------------------
// FLOATING POINT (review.md §4 Q36)
// ---------------------------------------------------------------------------
// This TU takes B31's per-file contraction flags (`-ffp-contract=off` /
// `--fmad=false`; see `src/bindings/blockAMR/CMakeLists.txt`), because the bar
// here is BITWISE against v1's numpy `wall_closure` and numpy cannot fuse. The
// flags apply to the TU, and `robin.H`'s arithmetic is inlined INTO this TU, so
// they are what makes the parity claim true — and they are what B32 must also
// take when it becomes the second includer. That requirement is stated in
// `robin.H`'s own header so it cannot be missed.

#include "robin.H"

#include "robin_data.H"
#include "wall_value.H"

#include <nanobind/nanobind.h>

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <array>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace
{

//! The closure, wrapped in the shape a wall functor has — NOT a wall treatment.
//!
//! It reads the closure ONE way per call, selected by `read`, so a `RecordSink`
//! row carries exactly one `(linear, constant)` pair and the four numbers plus
//! the third read are covered by the selector rather than by four sinks.
//!
//! Q34: it reads NO geometry. `alpha` and `beta` arrive through the real
//! `RobinView` (this is their first read anywhere — B30a shipped them in place
//! and unread), `gamma` through the real compiled `GammaExpr`, and `d` / `dg`
//! as arguments. In production `d` is the image point's distance from
//! `GhostCellData`, which is B32's seam; taking it as an argument here keeps
//! the parity suite free of geometry staging and lets a row choose the
//! ill-conditioned inputs the mutant coverage needs.
//!
//! `stencil_reach = 0` and this functor is never handed to `applyWall`. There
//! is no fab, no marker and no launch over a box: this TU tests a function.
struct WallClosureProbe
{
    static constexpr int stencil_reach = 0;

    ibm::RobinView robin;
    amrex::Real t;
    amrex::Real d;
    amrex::Real dg;
    int patch;
    int read; //!< 0 = value, 1 = grad, 2 = at(dg)

    template<class Sink>
    AMREX_GPU_HOST_DEVICE void operator()(int i, int j, int k, int n, Sink& sink) const
    {
        const ibm::WallClosure w =
            ibm::closure(robin.alpha[patch], robin.beta[patch], robin.gammaAt(patch, n, t), d);

        if (read == 0)
        {
            sink.linear(i, j, k, w.value_linear);
            sink.constant(w.value_constant);
        }
        else if (read == 1)
        {
            sink.linear(i, j, k, w.grad_linear);
            sink.constant(w.grad_constant);
        }
        else
        {
            sink.linear(i, j, k, w.atLinear(dg));
            sink.constant(w.atConstant(dg));
        }
    }
};

//! The probe conforms to the functor concept `wall_apply.H` documents, but it
//! is never handed to `applyWall` — so the one member nothing here reads is
//! read by this assertion instead of by a launch. Zero is the truth about the
//! closure: it is a function of its four arguments and touches no neighbour.
static_assert(
    WallClosureProbe::stencil_reach == 0, "the closure reads no cell but the one it is called on"
);

//! api §9: the hooks raise only about their own arguments, naming the value and
//! the bound. Nothing about the FORMULA raises — `den = 0` returns `+-inf` and
//! is not this function's business (review.md §4 Q43(c), `robin.H`'s header).
void requireProbeArgs(const char* fn, const ibm::RobinData& robin, int patch, int n, int read)
{
    if (patch < 0 || patch >= robin.npatch())
        throw std::runtime_error(
            std::string(fn) + ": patch " + std::to_string(patch) + " is outside the Robin table's "
            + std::to_string(robin.npatch())
        );
    if (n < 0 || n >= robin.ncomp())
        throw std::runtime_error(
            std::string(fn) + ": component " + std::to_string(n) + " is outside the Robin table's "
            + std::to_string(robin.ncomp())
        );
    if (read < 0 || read > 2)
        throw std::runtime_error(
            std::string(fn) + ": read " + std::to_string(read)
            + " is not one of 0 (the wall value), 1 (the wall gradient) or 2 (the field at the "
              "given ghost distance)"
        );
}

} // namespace

void registerRobinClosure(nb::module_& m)
{
    // TEST binding (api §4, §10.6) — tasks.md §3's verify column for B30b,
    // literally: the closure on a `RecordSink` CELL, host-side, read back as
    // the row v1's deleted row objects used to hand out.
    m.def(
        "_wall_closure_record",
        [](const ibm::RobinData& robin,
           int patch,
           int n,
           double t,
           double d,
           double dg,
           int read,
           int i,
           int j,
           int k) -> nb::tuple
        {
            requireProbeArgs("_wall_closure_record", robin, patch, n, read);

            const WallClosureProbe f {robin.view(), t, d, dg, patch, read};
            ibm::RecordSink rec;
            f(i, j, k, n, rec);

            if (rec.overflow)
                throw std::runtime_error(
                    "_wall_closure_record: the row emitted more than RecordSink::capacity = "
                    + std::to_string(ibm::RecordSink::capacity) + " linear entries"
                );

            nb::list entries;
            for (int e = 0; e < rec.count; ++e)
                entries.append(nb::make_tuple(
                    rec.entries[e].i, rec.entries[e].j, rec.entries[e].k, rec.entries[e].a
                ));
            return nb::make_tuple(entries, rec.c);
        },
        nb::arg("robin"),
        nb::arg("patch"),
        nb::arg("n"),
        nb::arg("t"),
        nb::arg("d"),
        nb::arg("dg"),
        nb::arg("read"),
        nb::arg("i") = 0,
        nb::arg("j") = 0,
        nb::arg("k") = 0,
        "TEST ONLY (B30b). robin.H's closure(alpha, beta, gamma(t), d) on the HOST at one cell "
        "against a RecordSink: returns ([(i, j, k, a), ...], c) for one reading of the closure "
        "— read=0 the wall value, read=1 the wall gradient, read=2 the field at the ghost "
        "distance dg. This is v1's wall_closure transcribed verbatim (review.md Q41); it makes "
        "no accuracy claim and guards no pole."
    );

    // TEST binding (api §4) — the SAME functor, in a one-thread kernel.
    //
    // Two things only this row can show. (i) `robin.H` is device-legal: a
    // `__host__ __device__` function that is never called from a kernel is
    // never codegen'd for the device, so without this hook B32 would be the
    // first place a device compile of the closure is attempted. (ii) host and
    // device agree BITWISE, which is a claim about the flags as much as about
    // the code.
    m.def(
        "_wall_closure_device",
        [](const ibm::RobinData& robin, int patch, int n, double t, double d, double dg, int read
        ) -> nb::tuple
        {
            requireProbeArgs("_wall_closure_device", robin, patch, n, read);

            amrex::Gpu::DeviceVector<amrex::Real> dv(2);
            amrex::Real* out = dv.data();
            const WallClosureProbe f {robin.view(), t, d, dg, patch, read};

            amrex::ParallelFor(
                1,
                [=] AMREX_GPU_DEVICE(int)
                {
                    ibm::RecordSink rec;
                    f(0, 0, 0, n, rec);
                    out[0] = rec.count > 0 ? rec.entries[0].a : 0.0;
                    out[1] = rec.c;
                }
            );
            amrex::Gpu::streamSynchronize();

            std::array<amrex::Real, 2> host {};
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, dv.begin(), dv.end(), host.begin());
            return nb::make_tuple(host[0], host[1]);
        },
        nb::arg("robin"),
        nb::arg("patch"),
        nb::arg("n"),
        nb::arg("t"),
        nb::arg("d"),
        nb::arg("dg"),
        nb::arg("read"),
        "TEST ONLY (B30b). The same closure reading as _wall_closure_record, computed by a "
        "one-thread device kernel and copied back: returns (a, c). It exists to force device "
        "codegen of robin.H in this session and to pin that host and device agree bitwise."
    );
}
