// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The wall frame's translation unit (B30a) — the formula-free half of B30.
//
// It compiles `wall_apply.H`, `wall_value.H` and `robin_data.H` against a
// CONFORMANCE HARNESS, `WallFrameProbe`, and exports the three things Python
// needs before a real pair exists: `WallMode`, `RobinData`, and two
// underscore-private test hooks.
//
// ---------------------------------------------------------------------------
// WHAT `WallFrameProbe` IS, AND WHAT IT IS EMPHATICALLY NOT
// ---------------------------------------------------------------------------
// It is NOT a wall treatment. It calls no closure, reads no alpha or beta,
// computes no distance and makes no accuracy claim. It exists to prove that the
// frame, the sinks, the guards and the compiled gamma(t) hold together — the
// verify column of tasks.md §3, "a functor frame is callable host-side against
// a RecordSink on one cell".
//
// The wall FORMULA is `robin.H`'s `closure(alpha, beta, gamma, d)` and it
// SHIPPED at **B30b** (review.md §4 Q41, the user decision: v1's `wall_closure`
// transcribed verbatim, 42/42 bitwise). The first `(operator, method)` pair to
// use it is `schemes/boundary/laplacian_ghost_cell.cpp` — `laplacian x
// ghostCell`, B32 — and that is where a wall treatment lives. Nothing in THIS
// file may be mistaken for one: the probe is never registered in
// `WALL_SCHEMES`, never becomes a pair, and its name cannot be read as
// `wall_<operator>_<method>`.
//
// ---------------------------------------------------------------------------
// FLOATING POINT (review.md §4 Q36)
// ---------------------------------------------------------------------------
// This TU takes NO per-file floating-point flags and therefore inherits AMReX's
// global `--use_fast_math` like every other binding source. That is a decision,
// not an oversight: nothing here is pinned bitwise against a numpy peer, so
// there is no association to protect. The exposure is bounded — the fast-math
// arms that change results (`--ftz`, `--prec-div`, `--prec-sqrt`, the `__cosf`
// family) are single-precision, and the only transcendentals in the frame are
// the f64 `std::cos`/`std::sin` of `GammaExpr::Harmonic`. The path that must be
// EXACT — a constant datum, verification §2's "exactly 0" probes — is the
// `Constant` tag, which evaluates no transcendental and performs no arithmetic
// at all.

#include "robin_data.H"
#include "wall_apply.H"
#include "wall_stage.H"
#include "wall_value.H"

#include "../../ibm/cell_type.H"
#include "../../ibm/geometry_view.H"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace nb = nanobind;

static_assert(
    std::is_same_v<std::int32_t, int>,
    "RobinData's gamma_form argument crosses the language boundary as numpy int32 and is read "
    "as `const int*`; on a platform where the two differ the binding would reinterpret it."
);

namespace
{

//! The conformance harness (NOT a wall treatment — see the file header).
//!
//! One `stencil_reach = 1` row: two linear entries on the x-axis neighbours,
//! and the patch's compiled datum through `constant`, alone. That is the whole
//! of it, and it is enough to exercise S1 (only WALL cells are written), S2
//! (the datum never arrives as a linear entry), S4 (the frame owns the mode),
//! S8 (the ghost-width guard) and R2 (`constant_scale = 0` drops exactly the
//! datum) without naming a boundary condition.
//!
//! Q34: the geometry is read at the target cell only — `patch(i, j, k)` and
//! nothing else. No `normal` or `wall_point` at a neighbour or a ghost index,
//! which is the question B29's freeze left open.
//!
//! S3 / Invariant F: this probe deliberately does NOT conform — its `i +- 1`
//! donors are emitted unconditionally, so at a WALL cell with a SOLID face
//! neighbour `ApplySink` reads a pinned cell. `laplacian x ghostCell`
//! (`laplacian_ghost_cell.cpp`, B32) gates each arm on
//! `m(ii, jj, kk) != ibm::SOLID`; this one does not, because it asserts nothing
//! about the answer, and `test_ibm_wall_functors.py`'s F-4 row holds the two
//! side by side at the same cell so the difference is measured rather than
//! narrated. What this probe exercises is the frame, and the frame is
//! indifferent to which cells a row names.
struct WallFrameProbe
{
    static constexpr int stencil_reach = 1;

    ibm::IbmGeometryView g;
    ibm::RobinView robin;
    amrex::Real t;
    amrex::GpuArray<amrex::Real, 3> dx;

    template<class Sink>
    AMREX_GPU_HOST_DEVICE void operator()(int i, int j, int k, int n, Sink& sink) const
    {
        const amrex::Real h2 = dx[0] * dx[0];
        sink.linear(i - 1, j, k, 1.0 / h2);
        sink.linear(i + 1, j, k, -1.0 / h2);
        sink.constant(robin.gammaAt(g.patch(i, j, k), n, t));
    }
};

//! The Maker: it holds what the frame refuses to know and hands the functor a
//! per-box view of it (design §4.1).
struct MakeWallFrameProbe
{
    using functor_type = WallFrameProbe;

    const ibm::IbmGeometryFab* g;
    ibm::RobinView robin;
    amrex::Real t;
    amrex::GpuArray<amrex::Real, 3> dx;

    WallFrameProbe operator()(const amrex::MFIter& mfi) const
    {
        return WallFrameProbe {ibm::makeGeometryView(*g, mfi), robin, t, dx};
    }

    //! S-5 (B30a-R): the two checks the frame has no types for. They used to
    //! sit in `_wall_frame_apply`'s lambda; here the frame makes them on every
    //! path that reaches `applyWall`, which is what the hook is for.
    void validate(const char* fn, int ncomp) const
    {
        ibm::requireGeometryLayout(*g, fn);
        ibm::requireRobinComponents(fn, robin.ncomp, ncomp);
    }
};

} // namespace

void registerWallFrame(nb::module_& m)
{
    nb::enum_<ibm::WallMode>(m, "WallMode", nb::is_arithmetic())
        .value("Overwrite", ibm::WallMode::Overwrite)
        .value("Add", ibm::WallMode::Add)
        .value("Assemble", ibm::WallMode::Assemble);

    m.attr("WALL_RECORD_CAPACITY") = ibm::RecordSink::capacity;
    m.attr("GAMMA_CONSTANT") = static_cast<int>(ibm::GammaExpr::Constant);
    m.attr("GAMMA_HARMONIC") = static_cast<int>(ibm::GammaExpr::Harmonic);

    using RoF64 = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
    using RoI32 = nb::ndarray<const std::int32_t, nb::c_contig, nb::device::cpu>;

    nb::class_<ibm::RobinData>(m, "RobinData")
        .def(
            "__init__",
            [](ibm::RobinData* self, RoF64 alpha, RoF64 beta, RoI32 gamma_form, RoF64 gamma_param)
            {
                if (alpha.ndim() != 1 || beta.ndim() != 1 || gamma_form.ndim() != 2
                    || gamma_param.ndim() != 3)
                    throw std::runtime_error(
                        "RobinData: the tables are alpha[npatch], beta[npatch], "
                        "gamma_form[npatch, ncomp] and gamma_param[npatch, ncomp, 4] — "
                        "one of those has the wrong number of dimensions"
                    );

                const std::size_t npatch = alpha.shape(0);
                const std::size_t ncomp = gamma_form.shape(1);
                const bool ok = beta.shape(0) == npatch && gamma_form.shape(0) == npatch
                             && gamma_param.shape(0) == npatch && gamma_param.shape(1) == ncomp
                             && gamma_param.shape(2) == 4;
                if (!ok)
                    throw std::runtime_error(
                        "RobinData: the four tables must agree on (npatch, ncomp) — alpha says "
                        "npatch = "
                        + std::to_string(npatch)
                        + " and gamma_form says ncomp = " + std::to_string(ncomp)
                        + ", so beta must be [npatch], gamma_form [npatch, ncomp] and "
                          "gamma_param [npatch, ncomp, 4]"
                    );

                new (self) ibm::RobinData(
                    alpha.data(),
                    beta.data(),
                    gamma_form.data(),
                    gamma_param.data(),
                    static_cast<int>(npatch),
                    static_cast<int>(ncomp)
                );
            },
            nb::arg("alpha"),
            nb::arg("beta"),
            nb::arg("gamma_form"),
            nb::arg("gamma_param"),
            "The per-patch (alpha, beta, gamma) a wall kernel reads (Q4). 'alpha' and 'beta' "
            "are float64 [npatch] — v1's SurfaceBC.robin() returns scalars for all three BC "
            "types and only gamma is ever per-component. 'gamma_form' is int32 [npatch, ncomp] "
            "(0 = Constant, 1 = Harmonic) and 'gamma_param' float64 [npatch, ncomp, 4] holding "
            "(a0, ac, as, omega), so that gamma(t) = a0 for Constant and "
            "a0 + ac*cos(omega t) + as*sin(omega t) for Harmonic. The patch index is the "
            "position of the body in sorted(mesh.bodies) — the same enumeration "
            "IbmGeometry.patch carries."
        )
        .def_prop_ro("npatch", &ibm::RobinData::npatch, "Patches the table is indexed by.")
        .def_prop_ro("ncomp", &ibm::RobinData::ncomp, "Field components gamma is carried for.");

    // TEST binding (api §4, §10.6) — tasks.md §3's verify column for B30a, and
    // read by nothing on an evaluate path. It calls the probe ON THE HOST for
    // ONE cell and returns the row the sink captured: the shape v1's deleted
    // row objects had, recovered from the shipped device code rather than from
    // a numpy builder beside it.
    m.def(
        "_wall_frame_record",
        [](const ibm::IbmGeometryFab& g,
           const ibm::RobinData& robin,
           const amrex::Geometry& geom,
           double t,
           int i,
           int j,
           int k,
           int n) -> nb::tuple
        {
            ibm::requireGeometryLayout(g, "_wall_frame_record");
            if (n < 0 || n >= robin.ncomp())
                throw std::runtime_error(
                    "_wall_frame_record: component " + std::to_string(n)
                    + " is outside the Robin table's " + std::to_string(robin.ncomp())
                );

            amrex::FArrayBox host;
            ibm::stageGeometryBox("_wall_frame_record", g, amrex::IntVect(i, j, k), host);
            const ibm::IbmGeometryView gv {host.const_array()};

            const int patch = gv.patch(i, j, k);
            if (patch < 0 || patch >= robin.npatch())
                throw std::runtime_error(
                    "_wall_frame_record: the geometry says cell [" + std::to_string(i) + ", "
                    + std::to_string(j) + ", " + std::to_string(k) + "] belongs to patch "
                    + std::to_string(patch) + ", but the Robin table has only "
                    + std::to_string(robin.npatch())
                );

            const WallFrameProbe f {gv, robin.view(), t, geom.CellSizeArray()};
            ibm::RecordSink rec;
            f(i, j, k, n, rec);

            if (rec.overflow)
                throw std::runtime_error(
                    "_wall_frame_record: the row emitted more than RecordSink::capacity = "
                    + std::to_string(ibm::RecordSink::capacity)
                    + " linear entries and was truncated; grow the sink"
                );

            nb::list entries;
            for (int e = 0; e < rec.count; ++e)
                entries.append(nb::make_tuple(
                    rec.entries[e].i, rec.entries[e].j, rec.entries[e].k, rec.entries[e].a
                ));
            return nb::make_tuple(entries, rec.c);
        },
        nb::arg("geom_ibm"),
        nb::arg("robin"),
        nb::arg("geom"),
        nb::arg("t"),
        nb::arg("i"),
        nb::arg("j"),
        nb::arg("k"),
        nb::arg("n"),
        "TEST ONLY (B30a). The wall frame's conformance probe, called on the HOST at one "
        "cell against a RecordSink: returns ([(i, j, k, a), ...], c) — the linear entries "
        "and the constant, which is the patch's gamma(t) and nothing else. This is not a "
        "wall treatment: it calls no closure and makes no accuracy claim."
    );

    // TEST binding (api §4) — the same probe through the real frame, over real
    // fabs: S1 (only WALL cells are written), the modes, `constant_scale`, and
    // the ghost-width guard. Argument order is the canonical twelve
    // (design §4.4) minus `method_data`, which the probe does not have.
    m.def(
        "_wall_frame_apply",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const ibm::CellTypeFab& ct,
           const ibm::IbmGeometryFab& g,
           const ibm::RobinData& robin,
           const amrex::Geometry& geom,
           double t,
           double coeff,
           int ncomp,
           ibm::WallMode mode,
           double constant_scale)
        {
            // No guard calls here: they are `MakeWallFrameProbe::validate`,
            // which `applyWall` calls once for every path into the sweep
            // (S-5). A binding that made them itself would be making them
            // twice, or — the failure mode S-5 exists for — in only one of the
            // several bindings a pair grows.
            ibm::applyWall(
                "_wall_frame_apply",
                out,
                phi,
                ct,
                MakeWallFrameProbe {&g, robin.view(), t, geom.CellSizeArray()},
                coeff,
                ncomp,
                mode,
                constant_scale
            );
            amrex::Gpu::streamSynchronize();
        },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("cell_type"),
        nb::arg("geom_ibm"),
        nb::arg("robin"),
        nb::arg("geom"),
        nb::arg("t"),
        nb::arg("coeff"),
        nb::arg("ncomp"),
        nb::arg("mode"),
        nb::arg("constant_scale"),
        "TEST ONLY (B30a). Runs the shared wall frame over every WALL cell with the "
        "conformance probe as its functor: out = coeff * (phi[i-1]/dx^2 - phi[i+1]/dx^2 + "
        "constant_scale * gamma(t)). Overwrite assigns, Add accumulates, Assemble raises. "
        "SOLID and FLUID cells are not written at all."
    );
}
