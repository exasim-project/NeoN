// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// `ghostCell`'s preprocessing, transcribed from `src/blockamr/ibm/ghost_cell.py`
// and `src/blockamr/ibm/geometry.py::_trilinear_donors` (B31).
//
// ---------------------------------------------------------------------------
// WHY THIS TU IS COMPILED WITH FLOATING-POINT CONTRACTION OFF
// ---------------------------------------------------------------------------
// Its acceptance bar is BITWISE equality with the numpy peer (tasks.md §3,
// review.md §4 Q29(d) — the ULP fallback is refused, a residual mismatch stays
// red). Every operation below is `+ - * / fabs floor max`: all correctly
// rounded in IEEE-754 binary64 on host and device alike, and none of them a
// libm call, so there is no transcendental freedom to lose. What is left is
// ASSOCIATION, and the two places the compiler would change it are the cell
// centre (`plo + (i + 0.5) * dx`) and the image point (`x + step * n`), both of
// which GCC (`-ffp-contract=fast`, its default) and nvcc (`--fmad=true`, its
// default, and re-asserted by AMReX's `--use_fast_math`) would contract into a
// single `fma` — one rounding where numpy does two.
//
// The mitigation is a per-source COMPILE_OPTIONS entry in this directory's
// `CMakeLists.txt` (`--fmad=false`, `-Xcompiler=-ffp-contract=off`), scoped to
// this file so no other TU's numerics move. Rejected alternatives, so they are
// not re-litigated: `#pragma STDC FP_CONTRACT OFF` (not honoured by GCC in C++
// mode), a `volatile` launder (an anti-optimization idiom in device code that
// hides the constraint from the build system), and accepting the FMA with a ULP
// tolerance (refused by Q29(d)).
//
// Each site the transcription must not "simplify" is marked H-a … H-h below.

#include "ghost_cell.H"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <AMReX_BoxArray.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_Scan.H>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace ibm
{

namespace
{

//! `[i, j, k]` — v1's `classify._cell_name`, so the raise reads identically.
std::string cellName(int i, int j, int k)
{
    return "[" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + "]";
}

} // namespace

void preprocess(
    GhostCellData& out,
    const CellTypeFab& ct,
    const IbmGeometryFab& g,
    const amrex::Geometry& geom,
    const std::vector<std::string>& patchNames
)
{
    requireGeometryLayout(g, "ghost_cell_preprocess");

    if (ct.boxArray() != g.boxArray())
        throw std::runtime_error(
            "ghost_cell_preprocess: the marker and the immersed-body geometry must share a "
            "BoxArray — the rows are selected from the marker and their geometry read from the "
            "same index"
        );

    const int ctNGrow = ct.nGrowVect().min();
    if (ctNGrow < 1)
        throw std::runtime_error(
            "ghost_cell_preprocess: Invariant F reads the marker at every live trilinear donor, "
            "and the half-cell image step puts a donor at most one cell outside the row's own "
            "cell, so the marker needs a ghost width of at least 1, but the CellTypeFab has "
            + std::to_string(ctNGrow) + "; grow the marker"
        );

    // `preprocess` itself reads the geometry only at VALID cells, so it needs no
    // geometry ghost of its own. What this checks is the pairing: `ct` must be
    // the marker `classifyDefault` produced from *this* `g`, and that pairing
    // requires geometry >= marker (geometry_view.H's ghost contract, half 1). A
    // caller who widened the marker and reused a narrow geometry gets the same
    // sentence here as at classification, rather than rows built against a
    // marker that was never validated over its own fab box.
    requireGeometryGhosts("ghost_cell_preprocess", g, ctNGrow);

    const amrex::GpuArray<amrex::Real, 3> dx = geom.CellSizeArray();
    const amrex::GpuArray<amrex::Real, 3> plo = geom.ProbLoArray();

    // -----------------------------------------------------------------------
    // Pass 1 — the row ORDER, which is the part a naive port gets wrong.
    //
    // v1 emits one row per fluid wall-layer cell, per local box in `MFIterator`
    // order, and within a box in `np.argwhere(depth == 1)` order — C order over
    // `(nx, ny, nz)`, i.e. sorted by i, then j, then k, with **k fastest**.
    // AMReX's natural linear index runs i fastest, and a `ParallelFor` plus an
    // atomic append would be a third order and a non-deterministic one. So each
    // WALL cell gets a deterministic rank from an exclusive scan over the box's
    // k-fastest linear index. The row order is a contract (B32 indexes this
    // data by the same rank), so the parity test below is also the order test.
    // -----------------------------------------------------------------------
    std::vector<amrex::Gpu::DeviceVector<int>> offsets;
    std::vector<int> rowBase;
    int total = 0;
    for (amrex::MFIter mfi(ct); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = mfi.validbox();
        const int n = static_cast<int>(bx.numPts());
        const auto blo = amrex::lbound(bx);
        const auto blen = amrex::length(bx);

        amrex::Gpu::DeviceVector<int> flag(static_cast<std::size_t>(n));
        amrex::Gpu::DeviceVector<int> offs(static_cast<std::size_t>(n));
        int* fp = flag.data();
        auto const& m = ct.const_array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                const int lin = ((i - blo.x) * blen.y + (j - blo.y)) * blen.z + (k - blo.z);
                fp[lin] = (m(i, j, k) == ibm::WALL) ? 1 : 0;
            }
        );
        amrex::Gpu::streamSynchronize();

        const int count = amrex::Scan::ExclusiveSum(n, flag.data(), offs.data());
        rowBase.push_back(total);
        offsets.push_back(std::move(offs));
        total += count;
    }

    out.nrows = total;
    const std::size_t nr = static_cast<std::size_t>(total);
    out.image_point.resize(nr * 3);
    out.donor.resize(nr * K * 3);
    out.weight.resize(nr * K);
    out.distance.resize(nr);

    // Scratch, not part of the data type: the cell and patch of each row, so
    // that an Invariant-F violation can be named host-side. Freed on return.
    amrex::Gpu::DeviceVector<int> rowCell(nr * 3);
    amrex::Gpu::DeviceVector<int> rowPatch(nr);

    // A device kernel cannot throw. The first violation in v1's own order —
    // `np.argwhere(bad)[0]`, i.e. the smallest `r * K + slot` — is claimed by an
    // atomic MIN, which makes the report both singular and deterministic.
    constexpr int NO_VIOLATION = std::numeric_limits<int>::max();
    amrex::Gpu::DeviceVector<int> bad(1);
    {
        const int none = NO_VIOLATION;
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, &none, &none + 1, bad.begin());
    }

    amrex::Real* ipp = out.image_point.data();
    int* dnp = out.donor.data();
    amrex::Real* wtp = out.weight.data();
    amrex::Real* dsp = out.distance.data();
    int* rcp = rowCell.data();
    int* rpp = rowPatch.data();
    int* badp = bad.data();

    // -----------------------------------------------------------------------
    // Pass 2 — one row per WALL cell. §3 of the plan, step by step.
    // -----------------------------------------------------------------------
    int b = 0;
    for (amrex::MFIter mfi(ct); mfi.isValid(); ++mfi, ++b)
    {
        const amrex::Box bx = mfi.validbox();
        const auto blo = amrex::lbound(bx);
        const auto blen = amrex::length(bx);
        const int base = rowBase[b];
        const int* offp = offsets[b].data();
        auto const& m = ct.const_array(mfi);
        const IbmGeometryView gv = makeGeometryView(g, mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (m(i, j, k) != ibm::WALL) return;
                const int lin = ((i - blo.x) * blen.y + (j - blo.y)) * blen.z + (k - blo.z);
                const int r = base + offp[lin];

                // Step 1 — the normal is READ out of the geometry, never
                // recomputed. No body sdf, no hypot, no sqrt, no atan2 is
                // evaluated anywhere in this kernel: that is what makes the
                // bitwise bar a statement about THIS arithmetic alone.
                const amrex::Real n0 = gv.normal(i, j, k, 0);
                const amrex::Real n1 = gv.normal(i, j, k, 1);
                const amrex::Real n2 = gv.normal(i, j, k, 2);

                // Steps 2-4 — `image_step` (ghost_cell.py:139-140).
                // H-d: divide PER COMPONENT, then max, then `0.5 / reach`. The
                // algebraically equal `0.5 * min_d(dx_d / |n_d|)` rounds
                // differently and loses the bit.
                const amrex::Real a0 = amrex::Math::abs(n0) / dx[0];
                const amrex::Real a1 = amrex::Math::abs(n1) / dx[1];
                const amrex::Real a2 = amrex::Math::abs(n2) / dx[2];
                amrex::Real reach = a0;
                if (a1 > reach) reach = a1;
                if (a2 > reach) reach = a2;
                const amrex::Real step = 0.5 / reach;

                // Step 5 — the cell centre (classify.py:149). `_index_coords`
                // wraps a periodic index; that wrap is a no-op here because
                // every row's cell is a VALID cell of the domain.
                // H-a: `plo + (i + 0.5) * dx` is TWO roundings. One `fma` is
                // one, and is a different number.
                const amrex::Real x0 = plo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
                const amrex::Real x1 = plo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
                const amrex::Real x2 = plo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];

                // Step 6 — the image point (ghost_cell.py:152). H-b: same
                // hazard, `x + step * n` must not become `fma(step, n, x)`.
                const amrex::Real p[3] = {x0 + step * n0, x1 + step * n1, x2 + step * n2};

                // Steps 7-9 — back to index space (geometry.py:207-209).
                // H-e, and this is the rewrite an optimizing author WILL find:
                // the round trip index -> coordinate -> index does NOT
                // short-circuit to `t_d = i_d + step * n_d / dx_d`. That
                // identity is exact in real arithmetic (the +0.5/-0.5 and the
                // prob_lo cancel) and wrong in floating point.
                // H-c: keep the divide BEFORE the `- 0.5`; the algebraically
                // equal `(p - plo - 0.5 * dx) / dx` is a different number.
                // H-g: keep the division — do not multiply by a reciprocal.
                int bidx[3];
                amrex::Real frac[3];
                for (int d = 0; d < 3; ++d)
                {
                    const amrex::Real t = (p[d] - plo[d]) / dx[d] - 0.5;
                    bidx[d] = static_cast<int>(std::floor(t));
                    frac[d] = t - static_cast<amrex::Real>(bidx[d]);
                }

                for (int s = 0; s < K; ++s)
                {
                    // Step 10 — slot s carries offset `o = (s>>2, s>>1, s) & 1`,
                    // which is geometry.py's `_OFFSETS` order.
                    const int o0 = (s >> 2) & 1;
                    const int o1 = (s >> 1) & 1;
                    const int o2 = s & 1;

                    // Step 11 (geometry.py:211-213)
                    const amrex::Real c0 = (o0 == 0) ? (1.0 - frac[0]) : frac[0];
                    const amrex::Real c1 = (o1 == 0) ? (1.0 - frac[1]) : frac[1];
                    const amrex::Real c2 = (o2 == 0) ? (1.0 - frac[2]) : frac[2];

                    // Step 12 — H-f: `(c0 * c1) * c2`, left to right. numpy's
                    // `corner.prod(axis=2)` reduces three contiguous elements
                    // sequentially (its pairwise machinery engages only above
                    // eight), so `c0 * (c1 * c2)`, or a `for (d) w *= c[d]` in
                    // another `d` order, changes the bits.
                    const amrex::Real w = (c0 * c1) * c2;

                    const int slot = r * K + s;
                    wtp[slot] = w;

                    // Step 15 — a dead slot (weight EXACTLY zero, not `< eps`)
                    // points at its own cell (ghost_cell.py:158-159), so it is
                    // inside every bound and is never a non-fluid read.
                    const bool live = (w != 0.0);
                    dnp[slot * 3 + 0] = live ? (bidx[0] + o0) : i;
                    dnp[slot * 3 + 1] = live ? (bidx[1] + o1) : j;
                    dnp[slot * 3 + 2] = live ? (bidx[2] + o2) : k;

                    // Step 14 — Invariant F on every live slot. v1 tests the
                    // analytic `min_b sdf_b > 0` at the (wrapped) donor centre,
                    // which under B28's classification is exactly "the donor's
                    // marker is not SOLID". Donors are emitted UNWRAPPED and
                    // reach at most one cell out, which is why the marker's
                    // ghost width is checked above.
                    if (live && m(bidx[0] + o0, bidx[1] + o1, bidx[2] + o2) == ibm::SOLID)
                    {
                        amrex::Gpu::Atomic::Min(badp, slot);
                    }
                }

                ipp[r * 3 + 0] = p[0];
                ipp[r * 3 + 1] = p[1];
                ipp[r * 3 + 2] = p[2];

                // Step 13 — the distance to the surface along n̂
                // (ghost_cell.py:164): the cell's own sdf plus the step.
                dsp[r] = gv.sdf(i, j, k) + step;

                rcp[r * 3 + 0] = i;
                rcp[r * 3 + 1] = j;
                rcp[r * 3 + 2] = k;
                rpp[r] = gv.patch(i, j, k);
            }
        );
    }
    amrex::Gpu::streamSynchronize();

    int badFlat = NO_VIOLATION;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, bad.begin(), bad.end(), &badFlat);
    if (badFlat == NO_VIOLATION) return;

    // Invariant F, host side: the sentence is copied from v1's
    // `_check_fluid_donors` (ghost_cell.py:179-185) word for word, because it
    // is what `test_a_non_fluid_donor_names_the_cell_and_the_patch` asserts.
    const int r = badFlat / K;
    int cell[3] = {0, 0, 0};
    int donorIdx[3] = {0, 0, 0};
    int patch = 0;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, rcp + 3 * r, rcp + 3 * r + 3, cell);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, rpp + r, rpp + r + 1, &patch);
    amrex::Gpu::copy(
        amrex::Gpu::deviceToHost, dnp + (badFlat * 3), dnp + (badFlat * 3) + 3, donorIdx
    );

    const std::string patchLabel =
        (patch >= 0 && static_cast<std::size_t>(patch) < patchNames.size())
            ? patchNames[static_cast<std::size_t>(patch)]
            : std::to_string(patch);

    throw std::runtime_error(
        "IBM band cell " + cellName(cell[0], cell[1], cell[2]) + " on patch '" + patchLabel
        + "' interpolates its image point from " + cellName(donorIdx[0], donorIdx[1], donorIdx[2])
        + ", which is not a fluid cell (Invariant F: a live stencil entry must be fluid, because "
          "a non-fluid cell holds the pin value and not data). The fluid on that side of the "
          "surface is under one cell deep there — refine the mesh or move the bodies apart."
    );
}

} // namespace ibm

namespace
{

//! A fresh HOST copy of a device vector, as a C-ordered numpy array owned by a
//! capsule. Never a pointer into device memory (band_table.cpp's precedent).
nb::object toNumpyReal(
    const amrex::Gpu::DeviceVector<amrex::Real>& v, std::size_t ndim, const std::size_t* shape
)
{
    const std::size_t n = v.size();
    auto* buf = new double[n ? n : 1];
    auto owner = nb::capsule(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    if (n) amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.end(), buf);
    return nb::cast(nb::ndarray<nb::numpy, double>(
        buf, ndim, shape, owner, nullptr, nb::dtype<double>(), nb::device::cpu::value, 0, 'C'
    ));
}

//! The same, as int32 — the dtype `test_ibm_ghost_cell.py:242` pins on v1's
//! `donor`, so the compiled side must export it too.
nb::object
toNumpyInt(const amrex::Gpu::DeviceVector<int>& v, std::size_t ndim, const std::size_t* shape)
{
    const std::size_t n = v.size();
    auto* buf = new std::int32_t[n ? n : 1];
    auto owner = nb::capsule(buf, [](void* p) noexcept { delete[] static_cast<std::int32_t*>(p); });
    if (n)
    {
        std::vector<int> host(n);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.end(), host.data());
        std::memcpy(buf, host.data(), n * sizeof(int));
    }
    return nb::cast(nb::ndarray<nb::numpy, std::int32_t>(
        buf, ndim, shape, owner, nullptr, nb::dtype<std::int32_t>(), nb::device::cpu::value, 0, 'C'
    ));
}

} // namespace

void registerGhostCell(nb::module_& m)
{
    m.attr("GHOST_CELL_K") = ibm::K;

    // TEST binding (api §4) — read by `test_ibm_ghost_cell_cpp.py`'s bitwise
    // parity section and by nothing on an evaluate path. Underscore-private and
    // a free function, so it never enters a bound class' vocabulary. Production
    // reaches the same rows as an opaque `GhostCellData` (api §10.3 item 4),
    // which is B36's wiring, not this task's.
    m.def(
        "_ghost_cell_numpy",
        [](ibm::CellTypeFab& ct,
           const ibm::IbmGeometryFab& g,
           const amrex::Geometry& geom,
           const std::vector<std::string>& patch_names)
        {
            ibm::GhostCellData d;
            ibm::preprocess(d, ct, g, geom, patch_names);

            const std::size_t n = static_cast<std::size_t>(d.nrows);
            const std::size_t kk = static_cast<std::size_t>(ibm::K);
            const std::size_t ipShape[2] = {n, 3};
            const std::size_t dnShape[3] = {n, kk, 3};
            const std::size_t wtShape[2] = {n, kk};
            const std::size_t dsShape[1] = {n};

            return nb::make_tuple(
                toNumpyReal(d.image_point, 2, ipShape),
                toNumpyInt(d.donor, 3, dnShape),
                toNumpyReal(d.weight, 2, wtShape),
                toNumpyReal(d.distance, 1, dsShape)
            );
        },
        nb::arg("ct"),
        nb::arg("geom_ibm"),
        nb::arg("geom"),
        nb::arg("patch_names"),
        "ghostCell's preprocessing, copied out to host numpy: (image_point (n, 3) f64, "
        "donor (n, 8, 3) int32, weight (n, 8) f64, distance (n,) f64). One row per WALL "
        "cell, per local box in MFIter order, k varying fastest within a box. "
        "'patch_names' is sorted(mesh.bodies); it names the patch in the Invariant-F "
        "raise and is otherwise unused."
    );
}
