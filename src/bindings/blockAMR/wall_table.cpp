// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// WallTable: the immersed-boundary row format (plans/IBM/ibm-row-format.md, v1).
// One row per non-fluid target cell,
//     phi(target) = sum_{k < ndonor} w_k * phi(donor_k) + b * gamma
// with a fixed donor stride K = 8. Rows are grouped per local box in MFIter
// order and addressed by the CSR-style box_offset array.
//
// The table is built in Python (numpy), copied once into this C++-owned handle
// and reused every evaluate until the grid changes — hence the grid_version
// staleness guard on both kernel entry points.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>

#include "bindings.hpp"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

// Fixed donor stride (trilinear). Entries k >= ndonor[r] are never read.
constexpr int kDonorStride = 8;


// ---------------------------------------------------------------------------
// numpy argument validation — every failure names the array and what was wrong.
// ---------------------------------------------------------------------------

// "(5, 8, 3)" for the array's actual shape.
static std::string shapeString(const nb::ndarray<nb::ro>& a)
{
    std::string s = "(";
    for (size_t i = 0; i < a.ndim(); ++i)
    {
        if (i > 0) s += ", ";
        s += std::to_string(a.shape(i));
    }
    s += ")";
    return s;
}

static void requireInt32(const nb::ndarray<nb::ro>& a, const char* name)
{
    if (a.dtype() != nb::dtype<int32_t>())
    {
        throw std::invalid_argument(std::string("WallTable: '") + name + "' must have dtype int32");
    }
}

static void requireFloat64(const nb::ndarray<nb::ro>& a, const char* name)
{
    if (a.dtype() != nb::dtype<double>())
    {
        throw std::invalid_argument(
            std::string("WallTable: '") + name + "' must have dtype float64"
        );
    }
}

// C-contiguous host memory: the constructor reads the buffer linearly.
static void requireCContiguousHost(const nb::ndarray<nb::ro>& a, const char* name)
{
    if (a.device_type() != nb::device::cpu::value)
    {
        throw std::invalid_argument(
            std::string("WallTable: '") + name + "' must be a CPU (numpy) array"
        );
    }
    // An empty array is vacuously contiguous, and numpy reports all-zero
    // strides for one (np.zeros((0, 3)).strides == (0, 0)) even though it sets
    // the C_CONTIGUOUS flag -- so the stride walk below would reject every
    // empty table. A body outside the domain produces exactly that.
    if (a.size() == 0)
    {
        return;
    }
    int64_t expected = 1;
    for (int i = static_cast<int>(a.ndim()) - 1; i >= 0; --i)
    {
        const size_t extent = a.shape(static_cast<size_t>(i));
        if (extent != 1 && a.stride(static_cast<size_t>(i)) != expected)
        {
            throw std::invalid_argument(
                std::string("WallTable: '") + name
                + "' must be C-contiguous (use numpy.ascontiguousarray)"
            );
        }
        expected *= static_cast<int64_t>(extent);
    }
}

static void
requireShape(const nb::ndarray<nb::ro>& a, const char* name, std::initializer_list<size_t> expected)
{
    bool ok = (a.ndim() == expected.size());
    if (ok)
    {
        size_t i = 0;
        for (size_t e : expected)
        {
            if (a.shape(i) != e)
            {
                ok = false;
                break;
            }
            ++i;
        }
    }
    if (!ok)
    {
        std::string want = "(";
        size_t i = 0;
        for (size_t e : expected)
        {
            if (i > 0) want += ", ";
            want += std::to_string(e);
            ++i;
        }
        want += ")";
        throw std::invalid_argument(
            std::string("WallTable: '") + name + "' must have shape " + want + ", got "
            + shapeString(a)
        );
    }
}

// Cast + upload one host buffer into a device-resident owned array.
template<typename Dst, typename Src>
static void uploadTo(amrex::Gpu::DeviceVector<Dst>& dst, const Src* src, size_t n)
{
    dst.resize(n);
    if (n == 0) return;
    std::vector<Dst> host(n);
    for (size_t i = 0; i < n; ++i)
    {
        host[i] = static_cast<Dst>(src[i]);
    }
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), dst.begin());
}


// ---------------------------------------------------------------------------
// The handle
// ---------------------------------------------------------------------------

// Device-copyable POD slice of a WallTable: raw pointers to one local box's
// contiguous run of rows. Captured by value into the ParallelFor lambdas.
// Only what a kernel dereferences lives here — nrows/ncomp are captured as
// locals by the launching loop, and `patch` is host-side diagnostics.
struct WallTableView
{
    const int* target;        // [nrows * 3]
    const int* donor;         // [nrows * K * 3]
    const amrex::Real* w;     // [nrows * K]
    const int* ndonor;        // [nrows]
    const amrex::Real* b;     // [nrows]
    const amrex::Real* gamma; // [nrows * ncomp]
};


// "(3, -1, 7)" for an index triple, for the bounds-check messages.
static std::string intVectString(const amrex::IntVect& v)
{
    std::string s = "(";
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        if (d > 0) s += ", ";
        s += std::to_string(v[d]);
    }
    s += ")";
    return s;
}


// Owns its arrays in Gpu::DeviceVector — the same arena (The_Arena) a MultiFab
// built with memory="default" uses, so the pointers are valid inside the
// ParallelFor kernels on every backend.
//
// box_offset, patch and the per-box index bounding boxes are deliberately kept
// in host memory: no kernel dereferences them. box_offset slices the rows per
// MFIter box, the bounding boxes drive the host-side bounds check, and patch is
// a diagnostic exposed to Python (row-format §2, "body/patch id").
class WallTable
{
public:

    WallTable(
        nb::ndarray<nb::ro> target,
        nb::ndarray<nb::ro> donor,
        nb::ndarray<nb::ro> w,
        nb::ndarray<nb::ro> ndonor,
        nb::ndarray<nb::ro> b,
        nb::ndarray<nb::ro> gamma,
        nb::ndarray<nb::ro> patch,
        nb::ndarray<nb::ro> box_offset,
        int grid_version
    );

    [[nodiscard]] int nrows() const { return m_nrows; }

    [[nodiscard]] int nbox() const { return m_nbox; }

    [[nodiscard]] int ncomp() const { return m_ncomp; }

    [[nodiscard]] int gridVersion() const { return m_grid_version; }

    [[nodiscard]] int rowBegin(int box) const { return m_box_offset[static_cast<size_t>(box)]; }

    [[nodiscard]] int rowEnd(int box) const { return m_box_offset[static_cast<size_t>(box) + 1]; }

    [[nodiscard]] const std::vector<int>& patch() const { return m_patch; }

    // Index bounding boxes of the rows of one local box, computed once in the
    // constructor. Empty runs report an inverted (lo > hi) range, which every
    // containment test passes trivially.
    [[nodiscard]] const amrex::IntVect& targetLo(int box) const
    {
        return m_target_lo[static_cast<size_t>(box)];
    }

    [[nodiscard]] const amrex::IntVect& targetHi(int box) const
    {
        return m_target_hi[static_cast<size_t>(box)];
    }

    [[nodiscard]] const amrex::IntVect& donorLo(int box) const
    {
        return m_donor_lo[static_cast<size_t>(box)];
    }

    [[nodiscard]] const amrex::IntVect& donorHi(int box) const
    {
        return m_donor_hi[static_cast<size_t>(box)];
    }

    // Pointers to the rows starting at row_begin.
    [[nodiscard]] WallTableView view(int row_begin) const
    {
        const size_t r0 = static_cast<size_t>(row_begin);
        WallTableView v;
        v.target = m_target.dataPtr() + r0 * 3;
        v.donor = m_donor.dataPtr() + r0 * kDonorStride * 3;
        v.w = m_w.dataPtr() + r0 * kDonorStride;
        v.ndonor = m_ndonor.dataPtr() + r0;
        v.b = m_b.dataPtr() + r0;
        v.gamma = m_gamma.dataPtr() + r0 * static_cast<size_t>(m_ncomp);
        return v;
    }

private:

    amrex::Gpu::DeviceVector<int> m_target;
    amrex::Gpu::DeviceVector<int> m_donor;
    amrex::Gpu::DeviceVector<amrex::Real> m_w;
    amrex::Gpu::DeviceVector<int> m_ndonor;
    amrex::Gpu::DeviceVector<amrex::Real> m_b;
    amrex::Gpu::DeviceVector<amrex::Real> m_gamma;
    // host only — see the class comment
    std::vector<int> m_patch;
    std::vector<int> m_box_offset;
    std::vector<amrex::IntVect> m_target_lo;
    std::vector<amrex::IntVect> m_target_hi;
    std::vector<amrex::IntVect> m_donor_lo;
    std::vector<amrex::IntVect> m_donor_hi;
    int m_nrows = 0;
    int m_nbox = 0;
    int m_ncomp = 0;
    int m_grid_version = 0;
};


WallTable::WallTable(
    nb::ndarray<nb::ro> target,
    nb::ndarray<nb::ro> donor,
    nb::ndarray<nb::ro> w,
    nb::ndarray<nb::ro> ndonor,
    nb::ndarray<nb::ro> b,
    nb::ndarray<nb::ro> gamma,
    nb::ndarray<nb::ro> patch,
    nb::ndarray<nb::ro> box_offset,
    int grid_version
)
{
    requireCContiguousHost(target, "target");
    requireCContiguousHost(donor, "donor");
    requireCContiguousHost(w, "w");
    requireCContiguousHost(ndonor, "ndonor");
    requireCContiguousHost(b, "b");
    requireCContiguousHost(gamma, "gamma");
    requireCContiguousHost(patch, "patch");
    requireCContiguousHost(box_offset, "box_offset");

    requireInt32(target, "target");
    requireInt32(donor, "donor");
    requireFloat64(w, "w");
    requireInt32(ndonor, "ndonor");
    requireFloat64(b, "b");
    requireFloat64(gamma, "gamma");
    requireInt32(patch, "patch");
    requireInt32(box_offset, "box_offset");

    // nrows comes from target; every other row array is checked against it.
    if (target.ndim() != 2 || target.shape(1) != 3)
    {
        throw std::invalid_argument(
            "WallTable: 'target' must have shape (nrows, 3), got " + shapeString(target)
        );
    }
    const size_t n = target.shape(0);

    requireShape(donor, "donor", {n, static_cast<size_t>(kDonorStride), 3});
    requireShape(w, "w", {n, static_cast<size_t>(kDonorStride)});
    requireShape(ndonor, "ndonor", {n});
    requireShape(b, "b", {n});
    requireShape(patch, "patch", {n});

    // ncomp comes from gamma.
    if (gamma.ndim() != 2 || gamma.shape(0) != n || gamma.shape(1) == 0)
    {
        throw std::invalid_argument(
            "WallTable: 'gamma' must have shape (" + std::to_string(n)
            + ", ncomp) with ncomp >= 1, got " + shapeString(gamma)
        );
    }
    const size_t ncomp = gamma.shape(1);

    // nbox comes from box_offset.
    if (box_offset.ndim() != 1 || box_offset.shape(0) < 1)
    {
        throw std::invalid_argument(
            "WallTable: 'box_offset' must have shape (nbox + 1,) with nbox >= 0, got "
            + shapeString(box_offset)
        );
    }
    const size_t nbox = box_offset.shape(0) - 1;

    // The kernels index donor[] and gamma[] without bounds checks, so the two
    // structural invariants are enforced here instead.
    const auto* ndonor_ptr = static_cast<const int32_t*>(ndonor.data());
    for (size_t r = 0; r < n; ++r)
    {
        if (ndonor_ptr[r] < 0 || ndonor_ptr[r] > kDonorStride)
        {
            throw std::invalid_argument(
                "WallTable: 'ndonor' must satisfy 0 <= ndonor <= " + std::to_string(kDonorStride)
                + ", got " + std::to_string(ndonor_ptr[r]) + " at row " + std::to_string(r)
            );
        }
    }

    const auto* offset_ptr = static_cast<const int32_t*>(box_offset.data());
    if (offset_ptr[0] != 0)
    {
        throw std::invalid_argument(
            "WallTable: 'box_offset' must start at 0, got " + std::to_string(offset_ptr[0])
        );
    }
    for (size_t i = 0; i < nbox; ++i)
    {
        if (offset_ptr[i + 1] < offset_ptr[i])
        {
            throw std::invalid_argument(
                "WallTable: 'box_offset' must be non-decreasing, got "
                + std::to_string(offset_ptr[i + 1]) + " after " + std::to_string(offset_ptr[i])
                + " at box " + std::to_string(i)
            );
        }
    }
    if (static_cast<size_t>(offset_ptr[nbox]) != n)
    {
        throw std::invalid_argument(
            "WallTable: 'box_offset' must end at nrows = " + std::to_string(n) + ", got "
            + std::to_string(offset_ptr[nbox])
        );
    }

    const auto* target_ptr = static_cast<const int32_t*>(target.data());
    const auto* donor_ptr = static_cast<const int32_t*>(donor.data());

    uploadTo(m_target, target_ptr, n * 3);
    uploadTo(m_donor, donor_ptr, n * kDonorStride * 3);
    uploadTo(m_w, static_cast<const double*>(w.data()), n * kDonorStride);
    uploadTo(m_ndonor, ndonor_ptr, n);
    uploadTo(m_b, static_cast<const double*>(b.data()), n);
    uploadTo(m_gamma, static_cast<const double*>(gamma.data()), n * ncomp);

    const auto* patch_ptr = static_cast<const int32_t*>(patch.data());
    m_patch.resize(n);
    for (size_t r = 0; r < n; ++r)
    {
        m_patch[r] = patch_ptr[r];
    }

    m_box_offset.resize(nbox + 1);
    for (size_t i = 0; i <= nbox; ++i)
    {
        m_box_offset[i] = offset_ptr[i];
    }

    // Per-box index bounding boxes. The kernels dereference the fab through
    // Array4, whose own index assert is compiled out of a release build
    // (AMReX_Array4.H, guarded by AMREX_DEBUG / AMREX_BOUND_CHECK), and this
    // handle never sees a Box — so the only place the ranges can be captured is
    // here, and they are checked against the MultiFab at every call.
    const amrex::IntVect empty_lo(std::numeric_limits<int>::max());
    const amrex::IntVect empty_hi(std::numeric_limits<int>::lowest());
    m_target_lo.assign(nbox, empty_lo);
    m_target_hi.assign(nbox, empty_hi);
    m_donor_lo.assign(nbox, empty_lo);
    m_donor_hi.assign(nbox, empty_hi);

    for (size_t ibox = 0; ibox < nbox; ++ibox)
    {
        const auto row_end = static_cast<size_t>(offset_ptr[ibox + 1]);
        for (size_t r = static_cast<size_t>(offset_ptr[ibox]); r < row_end; ++r)
        {
            const amrex::IntVect tgt(
                AMREX_D_DECL(target_ptr[r * 3], target_ptr[r * 3 + 1], target_ptr[r * 3 + 2])
            );
            m_target_lo[ibox].min(tgt);
            m_target_hi[ibox].max(tgt);

            // Entries k >= ndonor are never read, so they must not widen the box.
            const int nd = ndonor_ptr[r];
            for (int k = 0; k < nd; ++k)
            {
                const size_t o = (r * kDonorStride + static_cast<size_t>(k)) * 3;
                const amrex::IntVect dnr(
                    AMREX_D_DECL(donor_ptr[o], donor_ptr[o + 1], donor_ptr[o + 2])
                );
                m_donor_lo[ibox].min(dnr);
                m_donor_hi[ibox].max(dnr);
            }
        }
    }

    m_nrows = static_cast<int>(n);
    m_nbox = static_cast<int>(nbox);
    m_ncomp = static_cast<int>(ncomp);
    m_grid_version = grid_version;
}


// ---------------------------------------------------------------------------
// The two kernels
// ---------------------------------------------------------------------------

enum class RestrictMode
{
    Zero = 0,
    Overwrite = 1,
    AddSource = 2
};


// Staleness (ibm-row-format.md §3): a table built for a stale grid produces
// plausible wrong numbers, so this check is not optional and has no skip value.
static void checkCallArgs(
    const char* fn, const amrex::MultiFab& mf, const WallTable& tbl, int ncomp, int grid_version
)
{
    if (grid_version != tbl.gridVersion())
    {
        throw std::runtime_error(
            std::string(fn) + ": grid_version mismatch — the table was built for grid_version "
            + std::to_string(tbl.gridVersion()) + " but grid_version "
            + std::to_string(grid_version) + " was passed; the table is stale, rebuild it"
        );
    }
    const int mf_nbox = mf.local_size();
    if (mf_nbox != tbl.nbox())
    {
        throw std::runtime_error(
            std::string(fn) + ": nbox mismatch — the table has " + std::to_string(tbl.nbox())
            + " local boxes but the MultiFab has " + std::to_string(mf_nbox)
        );
    }
    // ncomp >= 1 is required even for RestrictMode::Zero, which never reads
    // gamma: a zero-component call is a caller bug in every mode, and one rule
    // for both kernels is easier to reason about than a mode-dependent one.
    if (ncomp < 1 || ncomp > tbl.ncomp())
    {
        throw std::invalid_argument(
            std::string(fn) + ": ncomp = " + std::to_string(ncomp)
            + " is outside the table's gamma range [1, " + std::to_string(tbl.ncomp()) + "]"
        );
    }
    if (ncomp > mf.nComp())
    {
        throw std::invalid_argument(
            std::string(fn) + ": ncomp = " + std::to_string(ncomp)
            + " exceeds the MultiFab's component count " + std::to_string(mf.nComp())
        );
    }
}


// Index-range guards (row-format §2 has no clause that bounds the donor stencil,
// and Array4's own assert is compiled out of a release build). Both run once per
// local box on the host: two IntVect comparisons, no measurable cost.

// A donor that lands outside the fab allocation is read anyway — silently wrong
// numbers, or an illegal address that only surfaces at some unrelated later
// sync. The builder's reach outside the valid box is not bounded by a constant
// in code (rows.py, "Halo width": 0-2 cells on cubic cells, 7 measured on a
// strongly anisotropic grid), so the field's ghost width is the thing that has
// to be big enough; this names the width it actually needs.
static void requireDonorsInFab(
    const char* fn,
    int local_box,
    const amrex::IntVect& lo,
    const amrex::IntVect& hi,
    const amrex::Box& fab_bx,
    const amrex::Box& valid_bx
)
{
    if (lo.allGE(fab_bx.smallEnd()) && hi.allLE(fab_bx.bigEnd())) return;

    int need = 0;
    int have = std::numeric_limits<int>::max();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        need = std::max(need, valid_bx.smallEnd(d) - lo[d]);
        need = std::max(need, hi[d] - valid_bx.bigEnd(d));
        have = std::min(have, valid_bx.smallEnd(d) - fab_bx.smallEnd(d));
        have = std::min(have, fab_bx.bigEnd(d) - valid_bx.bigEnd(d));
    }
    throw std::runtime_error(
        std::string(fn) + ": the donor indices of local box " + std::to_string(local_box) + " span "
        + intVectString(lo) + " .. " + intVectString(hi) + ", which reaches outside the fab box "
        + intVectString(fab_bx.smallEnd()) + " .. " + intVectString(fab_bx.bigEnd())
        + " (valid box " + intVectString(valid_bx.smallEnd()) + " .. "
        + intVectString(valid_bx.bigEnd()) + "); the geometry needs a ghost width of at least "
        + std::to_string(need) + " but the MultiFab has " + std::to_string(have)
    );
}


// Targets are written, so an out-of-range one corrupts memory rather than just
// reading it. A mismatch here means the table belongs to a different grid than
// the MultiFab, which the grid_version and nbox checks did not catch.
static void requireTargetsIn(
    const char* fn,
    const char* region,
    int local_box,
    const amrex::IntVect& lo,
    const amrex::IntVect& hi,
    const amrex::Box& allowed
)
{
    if (lo.allGE(allowed.smallEnd()) && hi.allLE(allowed.bigEnd())) return;

    throw std::runtime_error(
        std::string(fn) + ": the target indices of local box " + std::to_string(local_box)
        + " span " + intVectString(lo) + " .. " + intVectString(hi) + ", which reaches outside the "
        + region + " " + intVectString(allowed.smallEnd()) + " .. "
        + intVectString(allowed.bigEnd()) + "; the table does not belong to this MultiFab's grid"
    );
}


// P — apply the wall rows in place:
//   mf(target[r], n) = gamma_scale * b[r] * gamma[r, n]
//                    + sum_{k < ndonor[r]} w[r, k] * mf(donor[r, k], n)
// Race-free in place by Invariant D (targets and live donors are disjoint).
static void applyWallStencils(
    amrex::MultiFab& mf, const WallTable& tbl, int ncomp, amrex::Real gamma_scale, int grid_version
)
{
    checkCallArgs("apply_wall_stencils", mf, tbl, ncomp, grid_version);

    const int tncomp = tbl.ncomp();
    const int K = kDonorStride;
    int local_box = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const int lb = local_box++;
        const int row_begin = tbl.rowBegin(lb);
        const int nrows = tbl.rowEnd(lb) - row_begin;
        if (nrows <= 0) continue;

        const amrex::Box fab_bx = mfi.fabbox();
        const amrex::Box valid_bx = mfi.validbox();
        requireTargetsIn(
            "apply_wall_stencils", "valid box", lb, tbl.targetLo(lb), tbl.targetHi(lb), valid_bx
        );
        requireDonorsInFab(
            "apply_wall_stencils", lb, tbl.donorLo(lb), tbl.donorHi(lb), fab_bx, valid_bx
        );

        auto const& a = mf.array(mfi);
        const WallTableView t = tbl.view(row_begin);

        amrex::ParallelFor(
            nrows,
            [=] AMREX_GPU_DEVICE(int r)
            {
                const int* c = t.target + r * 3;
                const int nd = t.ndonor[r];
                for (int n = 0; n < ncomp; ++n)
                {
                    amrex::Real acc = gamma_scale * t.b[r] * t.gamma[r * tncomp + n];
                    for (int k = 0; k < nd; ++k)
                    {
                        const int* d = t.donor + (r * K + k) * 3;
                        acc += t.w[r * K + k] * a(d[0], d[1], d[2], n);
                    }
                    a(c[0], c[1], c[2], n) = acc;
                }
            }
        );
    }
}


// R — restrict the operator result in the band:
//   Zero      : out(target[r], n)  = 0
//   Overwrite : out(target[r], n)  = gamma_scale * b[r] * gamma[r, n]
//   AddSource : out(target[r], n) += gamma_scale * b[r] * gamma[r, n]
static void restrictBand(
    amrex::MultiFab& out,
    const WallTable& tbl,
    RestrictMode mode,
    int ncomp,
    amrex::Real gamma_scale,
    int grid_version
)
{
    checkCallArgs("restrict_band", out, tbl, ncomp, grid_version);

    const int tncomp = tbl.ncomp();
    const bool mode_zero = (mode == RestrictMode::Zero);
    const bool mode_add = (mode == RestrictMode::AddSource);

    int local_box = 0;
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const int lb = local_box++;
        const int row_begin = tbl.rowBegin(lb);
        const int nrows = tbl.rowEnd(lb) - row_begin;
        if (nrows <= 0) continue;

        // Only the targets are checked here: R reads no donors, and the result
        // MultiFab may legitimately carry fewer ghosts than the field the
        // donors were built for. Checked against the result's fab box because
        // that, not the valid box, is what the Array4 addresses.
        requireTargetsIn(
            "restrict_band", "result fab box", lb, tbl.targetLo(lb), tbl.targetHi(lb), mfi.fabbox()
        );

        auto const& a = out.array(mfi);
        const WallTableView t = tbl.view(row_begin);

        amrex::ParallelFor(
            nrows,
            [=] AMREX_GPU_DEVICE(int r)
            {
                const int* c = t.target + r * 3;
                for (int n = 0; n < ncomp; ++n)
                {
                    if (mode_zero)
                    {
                        a(c[0], c[1], c[2], n) = 0.0;
                    }
                    else
                    {
                        const amrex::Real src = gamma_scale * t.b[r] * t.gamma[r * tncomp + n];
                        if (mode_add) a(c[0], c[1], c[2], n) += src;
                        else
                            a(c[0], c[1], c[2], n) = src;
                    }
                }
            }
        );
    }
}


void registerWallTable(nb::module_& m)
{
    nb::enum_<RestrictMode>(m, "RestrictMode")
        .value("Zero", RestrictMode::Zero, "ghostCell: the operator result is meaningless there")
        .value("Overwrite", RestrictMode::Overwrite, "directForcing: write the body value")
        .value("AddSource", RestrictMode::AddSource, "penalization: add the forcing term");

    nb::class_<WallTable>(m, "WallTable")
        .def(
            "__init__",
            [](WallTable* self,
               nb::ndarray<nb::ro> target,
               nb::ndarray<nb::ro> donor,
               nb::ndarray<nb::ro> w,
               nb::ndarray<nb::ro> ndonor,
               nb::ndarray<nb::ro> b,
               nb::ndarray<nb::ro> gamma,
               nb::ndarray<nb::ro> patch,
               nb::ndarray<nb::ro> box_offset,
               int grid_version) {
                new (self)
                    WallTable(target, donor, w, ndonor, b, gamma, patch, box_offset, grid_version);
            },
            nb::arg("target"),
            nb::arg("donor"),
            nb::arg("w"),
            nb::arg("ndonor"),
            nb::arg("b"),
            nb::arg("gamma"),
            nb::arg("patch"),
            nb::arg("box_offset"),
            nb::arg("grid_version"),
            "Immersed-boundary wall rows (ibm-row-format.md v1). All arrays are "
            "C-contiguous numpy and are copied into the handle: target int32 [n, 3], "
            "donor int32 [n, 8, 3], w float64 [n, 8], ndonor int32 [n], b float64 [n], "
            "gamma float64 [n, ncomp], patch int32 [n], box_offset int32 [nbox + 1]."
        )
        .def_prop_ro("nrows", &WallTable::nrows, "Number of wall rows.")
        .def_prop_ro("nbox", &WallTable::nbox, "Number of local boxes the rows are grouped into.")
        .def_prop_ro("ncomp", &WallTable::ncomp, "Components carried by gamma.")
        .def_prop_ro(
            "grid_version",
            &WallTable::gridVersion,
            "The grid generation this table was built from."
        )
        .def_prop_ro(
            "patch",
            [](const WallTable& tbl)
            {
                const std::vector<int>& p = tbl.patch();
                auto* buf = new int32_t[p.size()];
                for (size_t r = 0; r < p.size(); ++r)
                {
                    buf[r] = static_cast<int32_t>(p[r]);
                }
                auto owner =
                    nb::capsule(buf, [](void* q) noexcept { delete[] static_cast<int32_t*>(q); });
                size_t shape[1] = {p.size()};
                return nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(
                    buf, 1, shape, owner, nullptr, nb::dtype<int32_t>(), nb::device::cpu::value, 0
                );
            },
            "The per-row body/patch id, as a fresh int32 numpy array of length nrows. "
            "Host-side diagnostics (per-patch forces); no kernel reads it."
        );

    m.def(
        "apply_wall_stencils",
        [](amrex::MultiFab& mf,
           const WallTable& table,
           int ncomp,
           double gamma_scale,
           int grid_version) { applyWallStencils(mf, table, ncomp, gamma_scale, grid_version); },
        nb::arg("mf"),
        nb::arg("table"),
        nb::arg("ncomp"),
        nb::arg("gamma_scale"),
        nb::arg("grid_version"),
        "P — apply the wall rows to mf in place: "
        "mf(target, n) = gamma_scale * b * gamma(r, n) + sum_k w(r, k) * mf(donor(r, k), n). "
        "gamma_scale = 1.0 is the affine apply, gamma_scale = 0.0 is P_lin alone (matvec mode). "
        "Raises RuntimeError if grid_version or the local box count does not match the table, or "
        "if the rows index outside mf (donors outside the fab box — the ghost width is too "
        "narrow — or targets outside the valid box)."
    );

    m.def(
        "restrict_band",
        [](amrex::MultiFab& out,
           const WallTable& table,
           RestrictMode mode,
           int ncomp,
           double gamma_scale,
           int grid_version) { restrictBand(out, table, mode, ncomp, gamma_scale, grid_version); },
        nb::arg("out"),
        nb::arg("table"),
        nb::arg("mode"),
        nb::arg("ncomp"),
        nb::arg("gamma_scale"),
        nb::arg("grid_version"),
        "R — restrict the operator result 'out' in the band: Zero writes 0, Overwrite writes "
        "gamma_scale * b * gamma(r, n), AddSource adds it. Never fused into the bulk kernel. "
        "Raises RuntimeError if grid_version, the local box count or the target index range "
        "does not match the table."
    );
}
