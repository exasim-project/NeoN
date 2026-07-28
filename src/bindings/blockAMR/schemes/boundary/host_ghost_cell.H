// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// A host-resident copy of `GhostCellData` for the row hooks (api §10.4) — B34.
//
// ---------------------------------------------------------------------------
// WHY THIS HEADER EXISTS, AND WHY IT IS NOT `wall_stage.H`
// ---------------------------------------------------------------------------
// `HostGhostCell` was `laplacian_ghost_cell.cpp`'s file-local helper (B32) and
// `div_ghost_cell.cpp` copied it verbatim (B33), which is exactly the shape api
// §10.4 names to avoid ("a private helper that a second file copies"). B33
// recorded the lift as owed at B34 — the third copy — and pointed it at
// `wall_stage.H`, beside the other staging helpers.
//
// **That destination is measured to be wrong, and it is a red test rather than
// a matter of taste.** `HostGhostCell` needs `ibm/ghost_cell.H`, which is one of
// the two headers whose arithmetic is pinned bitwise against numpy; and
// `test_ibm_laplacian_ghost_cell.py::test_every_includer_of_a_pinned_header_
// takes_the_fp_flags` walks the TRANSITIVE include closure of every `.cpp`
// under this tree and asserts `includers <= flagged` while asserting, on the
// next line, that `schemes/boundary/wall_frame.cpp` is NOT flagged.
// `wall_frame.cpp` includes `wall_stage.H`. Putting `HostGhostCell` there
// therefore makes `wall_frame.cpp` an includer of a pinned header that may not
// be flagged, and the only ways out are to flag it (Q36 refused that at B32 —
// its `GammaExpr::Harmonic` is a real fma site) or to weaken the row.
//
// So the lift lands in a header of its own, whose includer set is exactly the
// three wall pair TUs — all three already on the FP-flags list — and
// `wall_frame.cpp`'s include closure is untouched.
//
// ---------------------------------------------------------------------------
// WHAT IT IS FOR
// ---------------------------------------------------------------------------
// api §10.6 requires a wall functor to be callable ON THE HOST at one cell,
// against a `RecordSink`. `GhostCellData`'s four arrays are
// `Gpu::DeviceVector`s — device memory, not managed — so unlike `RobinView`
// they cannot simply be pointed at from the host. The values are the real ones
// and the code that reads them is the same `AMREX_GPU_HOST_DEVICE` member the
// kernel calls; only the residence changes. Nothing on an evaluate path
// constructs one.

#include "../../ibm/ghost_cell.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <vector>

namespace ibm
{

//! A host-resident copy of `GhostCellData`'s flat arrays, so a wall functor can
//! be called ON THE HOST at one cell (api §10.6).
struct HostGhostCell
{
    std::vector<amrex::Real> image_point;
    std::vector<int> donor;
    std::vector<amrex::Real> weight;
    std::vector<amrex::Real> distance;
    int nrows = 0;

    explicit HostGhostCell(const ibm::GhostCellData& d)
        : image_point(d.image_point.size()), donor(d.donor.size()), weight(d.weight.size()),
          distance(d.distance.size()), nrows(d.nrows)
    {
        if (d.nrows == 0) return;
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, d.image_point.begin(), d.image_point.end(), image_point.data()
        );
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d.donor.begin(), d.donor.end(), donor.data());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d.weight.begin(), d.weight.end(), weight.data());
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, d.distance.begin(), d.distance.end(), distance.data()
        );
        amrex::Gpu::streamSynchronize();
    }

    ibm::GhostCellView view() const
    {
        return ibm::GhostCellView {
            image_point.data(), donor.data(), weight.data(), distance.data(), nrows
        };
    }
};

} // namespace ibm
