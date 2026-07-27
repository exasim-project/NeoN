// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// BandTable: the immersed-boundary band row format (plans/IBM/row-contract.md, v1).
// One row per band cell,
//     out(target[r], n) = (or +=) sum_{k < nnz[r]} a[r,k] * phi(stencil[r,k], n)
//                                 + constant_scale * c[r,n]
// with a *runtime* stencil stride S (a property of the boundary scheme, not a
// global constant). Rows are grouped per local box in MFIter order and
// addressed by the CSR-style box_offset array — the same convention WallTable
// uses.
//
// The table is built in Python (numpy), copied once into this C++-owned handle
// and reused every evaluate until the IBM generation changes — hence the
// grid_version staleness guard on the kernel entry point.
//
// This is an additive translation unit: wall_table.cpp is untouched and still
// serves the old prolong/restrict path until the band path replaces it.

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


// ---------------------------------------------------------------------------
// numpy argument validation — every failure names the array and what was wrong.
// Deliberately a copy of wall_table.cpp's helpers rather than a shared header:
// they differ only in the message prefix, and the old table's error strings are
// asserted by the tests that still cover it.
// ---------------------------------------------------------------------------

// "(5, 8, 3)" for the array's actual shape.
static std::string bandShapeString(const nb::ndarray<nb::ro>& a)
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

static void bandRequireInt32(const nb::ndarray<nb::ro>& a, const char* name)
{
    if (a.dtype() != nb::dtype<int32_t>())
    {
        throw std::invalid_argument(std::string("BandTable: '") + name + "' must have dtype int32");
    }
}

static void bandRequireFloat64(const nb::ndarray<nb::ro>& a, const char* name)
{
    if (a.dtype() != nb::dtype<double>())
    {
        throw std::invalid_argument(
            std::string("BandTable: '") + name + "' must have dtype float64"
        );
    }
}

// C-contiguous host memory: the constructor reads the buffer linearly.
static void bandRequireCContiguousHost(const nb::ndarray<nb::ro>& a, const char* name)
{
    if (a.device_type() != nb::device::cpu::value)
    {
        throw std::invalid_argument(
            std::string("BandTable: '") + name + "' must be a CPU (numpy) array"
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
                std::string("BandTable: '") + name
                + "' must be C-contiguous (use numpy.ascontiguousarray)"
            );
        }
        expected *= static_cast<int64_t>(extent);
    }
}

static void bandRequireShape(
    const nb::ndarray<nb::ro>& a, const char* name, std::initializer_list<size_t> expected
)
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
            std::string("BandTable: '") + name + "' must have shape " + want + ", got "
            + bandShapeString(a)
        );
    }
}

// Cast + upload one host buffer into a device-resident owned array.
template<typename Dst, typename Src>
static void bandUploadTo(amrex::Gpu::DeviceVector<Dst>& dst, const Src* src, size_t n)
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


// "(3, -1, 7)" for an index triple, for the bounds-check messages.
static std::string bandIntVectString(const amrex::IntVect& v)
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


// ---------------------------------------------------------------------------
// The handle
// ---------------------------------------------------------------------------

// Device-copyable POD slice of a BandTable: raw pointers to one local box's
// contiguous run of rows. Captured by value into the ParallelFor lambda.
// Only what the kernel dereferences lives here — nrows/ncomp/stride are
// captured as locals by the launching loop, and `patch` is host-side
// diagnostics.
struct BandTableView
{
    const int* target;    // [nrows * 3]
    const int* stencil;   // [nrows * S * 3]
    const amrex::Real* a; // [nrows * S]
    const int* nnz;       // [nrows]
    const amrex::Real* c; // [nrows * ncomp]
};


// Owns its arrays in Gpu::DeviceVector — the same arena (The_Arena) a MultiFab
// built with memory="default" uses, so the pointers are valid inside the
// ParallelFor kernel on every backend.
//
// box_offset, patch and the per-box index bounding boxes are deliberately kept
// in host memory: no kernel dereferences them. box_offset slices the rows per
// MFIter box, the bounding boxes drive the host-side bounds checks, and patch
// is a diagnostic exposed to Python (row-contract §2) that nothing consumes
// yet — per-patch forces are the first reader.
class BandTable
{
public:

    BandTable(
        nb::ndarray<nb::ro> target,
        nb::ndarray<nb::ro> stencil,
        nb::ndarray<nb::ro> a,
        nb::ndarray<nb::ro> nnz,
        nb::ndarray<nb::ro> c,
        nb::ndarray<nb::ro> patch,
        nb::ndarray<nb::ro> box_offset,
        int grid_version
    );

    [[nodiscard]] int nrows() const { return m_nrows; }

    [[nodiscard]] int nbox() const { return m_nbox; }

    [[nodiscard]] int ncomp() const { return m_ncomp; }

    [[nodiscard]] int stride() const { return m_stride; }

    [[nodiscard]] int gridVersion() const { return m_grid_version; }

    // The widest live stencil in the table. Zero means no row reads anything,
    // which is what makes an in-place apply legal (row-contract §7).
    [[nodiscard]] int maxNnz() const { return m_max_nnz; }

    [[nodiscard]] const std::vector<int>& patch() const { return m_patch; }

    [[nodiscard]] int rowBegin(int box) const { return m_box_offset[static_cast<size_t>(box)]; }

    [[nodiscard]] int rowEnd(int box) const { return m_box_offset[static_cast<size_t>(box) + 1]; }

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

    [[nodiscard]] const amrex::IntVect& stencilLo(int box) const
    {
        return m_stencil_lo[static_cast<size_t>(box)];
    }

    [[nodiscard]] const amrex::IntVect& stencilHi(int box) const
    {
        return m_stencil_hi[static_cast<size_t>(box)];
    }

    // Pointers to the rows starting at row_begin.
    [[nodiscard]] BandTableView view(int row_begin) const
    {
        const size_t r0 = static_cast<size_t>(row_begin);
        const size_t s = static_cast<size_t>(m_stride);
        BandTableView v;
        v.target = m_target.dataPtr() + r0 * 3;
        v.stencil = m_stencil.dataPtr() + r0 * s * 3;
        v.a = m_a.dataPtr() + r0 * s;
        v.nnz = m_nnz.dataPtr() + r0;
        v.c = m_c.dataPtr() + r0 * static_cast<size_t>(m_ncomp);
        return v;
    }

private:

    amrex::Gpu::DeviceVector<int> m_target;
    amrex::Gpu::DeviceVector<int> m_stencil;
    amrex::Gpu::DeviceVector<amrex::Real> m_a;
    amrex::Gpu::DeviceVector<int> m_nnz;
    amrex::Gpu::DeviceVector<amrex::Real> m_c;
    // host only — see the class comment
    std::vector<int> m_patch;
    std::vector<int> m_box_offset;
    std::vector<amrex::IntVect> m_target_lo;
    std::vector<amrex::IntVect> m_target_hi;
    std::vector<amrex::IntVect> m_stencil_lo;
    std::vector<amrex::IntVect> m_stencil_hi;
    int m_nrows = 0;
    int m_nbox = 0;
    int m_ncomp = 0;
    int m_stride = 0;
    int m_max_nnz = 0;
    int m_grid_version = 0;
};


BandTable::BandTable(
    nb::ndarray<nb::ro> target,
    nb::ndarray<nb::ro> stencil,
    nb::ndarray<nb::ro> a,
    nb::ndarray<nb::ro> nnz,
    nb::ndarray<nb::ro> c,
    nb::ndarray<nb::ro> patch,
    nb::ndarray<nb::ro> box_offset,
    int grid_version
)
{
    bandRequireCContiguousHost(target, "target");
    bandRequireCContiguousHost(stencil, "stencil");
    bandRequireCContiguousHost(a, "a");
    bandRequireCContiguousHost(nnz, "nnz");
    bandRequireCContiguousHost(c, "c");
    bandRequireCContiguousHost(patch, "patch");
    bandRequireCContiguousHost(box_offset, "box_offset");

    bandRequireInt32(target, "target");
    bandRequireInt32(stencil, "stencil");
    bandRequireFloat64(a, "a");
    bandRequireInt32(nnz, "nnz");
    bandRequireFloat64(c, "c");
    bandRequireInt32(patch, "patch");
    bandRequireInt32(box_offset, "box_offset");

    // nrows comes from target; every other row array is checked against it.
    if (target.ndim() != 2 || target.shape(1) != 3)
    {
        throw std::invalid_argument(
            "BandTable: 'target' must have shape (nrows, 3), got " + bandShapeString(target)
        );
    }
    const size_t n = target.shape(0);

    // The stencil stride is a property of the boundary scheme, so it is read
    // off the array rather than being a compile-time constant.
    if (stencil.ndim() != 3 || stencil.shape(0) != n || stencil.shape(2) != 3)
    {
        throw std::invalid_argument(
            "BandTable: 'stencil' must have shape (" + std::to_string(n) + ", stride, 3), got "
            + bandShapeString(stencil)
        );
    }
    const size_t stride = stencil.shape(1);

    bandRequireShape(a, "a", {n, stride});
    bandRequireShape(nnz, "nnz", {n});
    bandRequireShape(patch, "patch", {n});

    // ncomp comes from c.
    if (c.ndim() != 2 || c.shape(0) != n || c.shape(1) == 0)
    {
        throw std::invalid_argument(
            "BandTable: 'c' must have shape (" + std::to_string(n)
            + ", ncomp) with ncomp >= 1, got " + bandShapeString(c)
        );
    }
    const size_t ncomp = c.shape(1);

    // nbox comes from box_offset.
    if (box_offset.ndim() != 1 || box_offset.shape(0) < 1)
    {
        throw std::invalid_argument(
            "BandTable: 'box_offset' must have shape (nbox + 1,) with nbox >= 0, got "
            + bandShapeString(box_offset)
        );
    }
    const size_t nbox = box_offset.shape(0) - 1;

    // The kernel indexes stencil[], a[] and c[] without bounds checks, so the
    // structural invariants are enforced here instead.
    const auto* nnz_ptr = static_cast<const int32_t*>(nnz.data());
    int max_nnz = 0;
    for (size_t r = 0; r < n; ++r)
    {
        if (nnz_ptr[r] < 0 || static_cast<size_t>(nnz_ptr[r]) > stride)
        {
            throw std::invalid_argument(
                "BandTable: 'nnz' must satisfy 0 <= nnz <= stride = " + std::to_string(stride)
                + ", got " + std::to_string(nnz_ptr[r]) + " at row " + std::to_string(r)
            );
        }
        max_nnz = std::max(max_nnz, static_cast<int>(nnz_ptr[r]));
    }

    const auto* offset_ptr = static_cast<const int32_t*>(box_offset.data());
    if (offset_ptr[0] != 0)
    {
        throw std::invalid_argument(
            "BandTable: 'box_offset' must start at 0, got " + std::to_string(offset_ptr[0])
        );
    }
    for (size_t i = 0; i < nbox; ++i)
    {
        if (offset_ptr[i + 1] < offset_ptr[i])
        {
            throw std::invalid_argument(
                "BandTable: 'box_offset' must be non-decreasing, got "
                + std::to_string(offset_ptr[i + 1]) + " after " + std::to_string(offset_ptr[i])
                + " at box " + std::to_string(i)
            );
        }
    }
    if (static_cast<size_t>(offset_ptr[nbox]) != n)
    {
        throw std::invalid_argument(
            "BandTable: 'box_offset' must end at nrows = " + std::to_string(n) + ", got "
            + std::to_string(offset_ptr[nbox])
        );
    }

    const auto* target_ptr = static_cast<const int32_t*>(target.data());
    const auto* stencil_ptr = static_cast<const int32_t*>(stencil.data());

    bandUploadTo(m_target, target_ptr, n * 3);
    bandUploadTo(m_stencil, stencil_ptr, n * stride * 3);
    bandUploadTo(m_a, static_cast<const double*>(a.data()), n * stride);
    bandUploadTo(m_nnz, nnz_ptr, n);
    bandUploadTo(m_c, static_cast<const double*>(c.data()), n * ncomp);

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

    // Per-box index bounding boxes. The kernel dereferences the fabs through
    // Array4, whose own index assert is compiled out of a release build
    // (AMReX_Array4.H, guarded by AMREX_DEBUG / AMREX_BOUND_CHECK), and this
    // handle never sees a Box — so the only place the ranges can be captured is
    // here, and they are checked against the MultiFabs at every call.
    const amrex::IntVect empty_lo(std::numeric_limits<int>::max());
    const amrex::IntVect empty_hi(std::numeric_limits<int>::lowest());
    m_target_lo.assign(nbox, empty_lo);
    m_target_hi.assign(nbox, empty_hi);
    m_stencil_lo.assign(nbox, empty_lo);
    m_stencil_hi.assign(nbox, empty_hi);

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

            // Entries k >= nnz are never read, so they must not widen the box.
            const int live = nnz_ptr[r];
            for (int k = 0; k < live; ++k)
            {
                const size_t o = (r * stride + static_cast<size_t>(k)) * 3;
                const amrex::IntVect s(
                    AMREX_D_DECL(stencil_ptr[o], stencil_ptr[o + 1], stencil_ptr[o + 2])
                );
                m_stencil_lo[ibox].min(s);
                m_stencil_hi[ibox].max(s);
            }
        }
    }

    m_nrows = static_cast<int>(n);
    m_nbox = static_cast<int>(nbox);
    m_ncomp = static_cast<int>(ncomp);
    m_stride = static_cast<int>(stride);
    m_max_nnz = max_nnz;
    m_grid_version = grid_version;
}


// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

enum class BandMode
{
    Overwrite = 0,
    Add = 1
};


// Staleness (row-contract §5): a table built for a stale IBM generation
// produces plausible wrong numbers, so this check is not optional and has no
// skip value. Everything else here is an agreement check between the table and
// the two MultiFabs it is about to index.
static void checkBandCallArgs(
    const char* fn,
    const amrex::MultiFab& out,
    const amrex::MultiFab& phi,
    const BandTable& tbl,
    int ncomp,
    int grid_version
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
    // The two-fabs rule forbids in-place *reads*; a table whose every row has
    // nnz = 0 has none, and that is exactly the non-fluid pin (row-contract §7).
    if (&out == &phi && tbl.maxNnz() > 0)
    {
        throw std::runtime_error(
            std::string(fn)
            + ": 'out' and 'phi' are the same MultiFab, but the table's widest "
              "row reads "
            + std::to_string(tbl.maxNnz())
            + " cells; an in-place apply is only defined for a table whose rows read nothing "
              "(every nnz = 0, the non-fluid pin)"
        );
    }
    // One MFIter walks both fabs (it is built on phi and indexes out by local
    // index), so they have to be the same grid on the same ranks.
    if (out.boxArray() != phi.boxArray() || out.DistributionMap() != phi.DistributionMap())
    {
        throw std::runtime_error(
            std::string(fn)
            + ": 'out' and 'phi' are built on different grids; the band rows "
              "address one BoxArray and one DistributionMapping"
        );
    }
    const int phi_nbox = phi.local_size();
    if (phi_nbox != tbl.nbox())
    {
        throw std::runtime_error(
            std::string(fn) + ": nbox mismatch — the table has " + std::to_string(tbl.nbox())
            + " local boxes but the MultiFab has " + std::to_string(phi_nbox)
        );
    }
    // ncomp >= 1 is required even for a table whose rows are all constant: a
    // zero-component call is a caller bug in every mode.
    if (ncomp < 1 || ncomp > tbl.ncomp())
    {
        throw std::invalid_argument(
            std::string(fn) + ": ncomp = " + std::to_string(ncomp)
            + " is outside the table's c range [1, " + std::to_string(tbl.ncomp()) + "]"
        );
    }
    if (ncomp > phi.nComp() || ncomp > out.nComp())
    {
        throw std::invalid_argument(
            std::string(fn) + ": ncomp = " + std::to_string(ncomp)
            + " exceeds the component count of phi (" + std::to_string(phi.nComp()) + ") or out ("
            + std::to_string(out.nComp()) + ")"
        );
    }
}


// Index-range guards (the row contract has no clause that bounds the stencil,
// and Array4's own assert is compiled out of a release build). Both run once
// per local box on the host: two IntVect comparisons, no measurable cost.

// A stencil entry that lands outside the fab allocation is read anyway —
// silently wrong numbers, or an illegal address that only surfaces at some
// unrelated later sync. The row builder's reach outside the valid box is not
// bounded by a constant in code, so the field's ghost width is the thing that
// has to be big enough; this names the width it actually needs.
static void requireStencilsInFab(
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
        std::string(fn) + ": the stencil indices of local box " + std::to_string(local_box)
        + " span " + bandIntVectString(lo) + " .. " + bandIntVectString(hi)
        + ", which reaches outside the fab box " + bandIntVectString(fab_bx.smallEnd()) + " .. "
        + bandIntVectString(fab_bx.bigEnd()) + " (valid box "
        + bandIntVectString(valid_bx.smallEnd()) + " .. " + bandIntVectString(valid_bx.bigEnd())
        + "); the geometry needs a ghost width of at least " + std::to_string(need)
        + " but the MultiFab has " + std::to_string(have)
    );
}


// Targets are written, so an out-of-range one corrupts memory rather than just
// reading it. A mismatch here means the table belongs to a different grid than
// the MultiFab, which the grid_version and nbox checks did not catch — or that
// the rows are not in MFIter order.
static void requireBandTargetsIn(
    const char* fn,
    int local_box,
    const amrex::IntVect& lo,
    const amrex::IntVect& hi,
    const amrex::Box& valid_bx
)
{
    if (lo.allGE(valid_bx.smallEnd()) && hi.allLE(valid_bx.bigEnd())) return;

    throw std::runtime_error(
        std::string(fn) + ": the target indices of local box " + std::to_string(local_box)
        + " span " + bandIntVectString(lo) + " .. " + bandIntVectString(hi)
        + ", which reaches outside the valid box " + bandIntVectString(valid_bx.smallEnd()) + " .. "
        + bandIntVectString(valid_bx.bigEnd())
        + "; the table does not belong to this MultiFab's grid, or its rows are not grouped in "
          "MFIter order"
    );
}


// The band sweep:
//   out(target[r], n)  = sum_{k < nnz[r]} a[r,k] * phi(stencil[r,k], n)
//                        + constant_scale * c[r, n]        (Overwrite)
//   out(target[r], n) += the same                          (Add)
// phi is read, out is written, and they are different MultiFabs — so there is
// no in-place race to design around and no ordering constraint between rows.
static void applyBandRows(
    amrex::MultiFab& out,
    const amrex::MultiFab& phi,
    const BandTable& tbl,
    int ncomp,
    BandMode mode,
    amrex::Real constant_scale,
    int grid_version
)
{
    checkBandCallArgs("apply_band_rows", out, phi, tbl, ncomp, grid_version);

    const int tncomp = tbl.ncomp();
    const int stride = tbl.stride();
    const bool add = (mode == BandMode::Add);

    int local_box = 0;
    // Iterated over phi so that fabbox() is the *read* fab's grown box, which
    // is what the stencil has to fit into; validbox() is shared with out, whose
    // BoxArray was checked to match.
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        const int lb = local_box++;
        const int row_begin = tbl.rowBegin(lb);
        const int nrows = tbl.rowEnd(lb) - row_begin;
        if (nrows <= 0) continue;

        const amrex::Box fab_bx = mfi.fabbox();
        const amrex::Box valid_bx = mfi.validbox();
        requireBandTargetsIn("apply_band_rows", lb, tbl.targetLo(lb), tbl.targetHi(lb), valid_bx);
        requireStencilsInFab(
            "apply_band_rows", lb, tbl.stencilLo(lb), tbl.stencilHi(lb), fab_bx, valid_bx
        );

        auto const& src = phi.const_array(mfi);
        auto const& dst = out.array(mfi);
        const BandTableView t = tbl.view(row_begin);

        amrex::ParallelFor(
            nrows,
            [=] AMREX_GPU_DEVICE(int r)
            {
                const int* tc = t.target + r * 3;
                const int live = t.nnz[r];
                for (int n = 0; n < ncomp; ++n)
                {
                    amrex::Real acc = constant_scale * t.c[r * tncomp + n];
                    for (int k = 0; k < live; ++k)
                    {
                        const int* s = t.stencil + (r * stride + k) * 3;
                        acc += t.a[r * stride + k] * src(s[0], s[1], s[2], n);
                    }
                    if (add) dst(tc[0], tc[1], tc[2], n) += acc;
                    else
                        dst(tc[0], tc[1], tc[2], n) = acc;
                }
            }
        );
    }
}


void registerBandTable(nb::module_& m)
{
    nb::enum_<BandMode>(m, "BandMode")
        .value("Overwrite", BandMode::Overwrite, "replace what the interior sweep wrote")
        .value("Add", BandMode::Add, "add to what the interior sweep wrote");

    nb::class_<BandTable>(m, "BandTable")
        .def(
            "__init__",
            [](BandTable* self,
               nb::ndarray<nb::ro> target,
               nb::ndarray<nb::ro> stencil,
               nb::ndarray<nb::ro> a,
               nb::ndarray<nb::ro> nnz,
               nb::ndarray<nb::ro> c,
               nb::ndarray<nb::ro> patch,
               nb::ndarray<nb::ro> box_offset,
               int grid_version)
            { new (self) BandTable(target, stencil, a, nnz, c, patch, box_offset, grid_version); },
            nb::arg("target"),
            nb::arg("stencil"),
            nb::arg("a"),
            nb::arg("nnz"),
            nb::arg("c"),
            nb::arg("patch"),
            nb::arg("box_offset"),
            nb::arg("grid_version"),
            "Immersed-boundary band rows (row-contract.md v1). All arrays are C-contiguous "
            "numpy and are copied into the handle: target int32 [n, 3], stencil int32 "
            "[n, stride, 3], a float64 [n, stride], nnz int32 [n], c float64 [n, ncomp], "
            "patch int32 [n], box_offset int32 [nbox + 1]. The stencil stride is read off "
            "'stencil' — it is a property of the boundary scheme, not a global constant."
        )
        .def_prop_ro("nrows", &BandTable::nrows, "Number of band rows.")
        .def_prop_ro("nbox", &BandTable::nbox, "Number of local boxes the rows are grouped into.")
        .def_prop_ro("ncomp", &BandTable::ncomp, "Components carried by c.")
        .def_prop_ro("stride", &BandTable::stride, "Stencil stride S: the row's slot count.")
        .def_prop_ro(
            "grid_version", &BandTable::gridVersion, "The IBM generation this table was built from."
        )
        .def_prop_ro(
            "max_nnz",
            &BandTable::maxNnz,
            "The widest live stencil in the table. Zero means no row reads a cell, which is "
            "what makes an in-place apply (out is phi) legal."
        )
        .def_prop_ro(
            "patch",
            [](const BandTable& tbl)
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
            // The ndarray owns its buffer through the capsule, so the property
            // must not use def_prop_ro's default reference_internal policy.
            nb::rv_policy::move,
            "The per-row body/patch id, as a fresh int32 numpy array of length nrows. "
            "Host-side diagnostics (per-patch forces); no kernel reads it."
        );

    m.def(
        "apply_band_rows",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const BandTable& table,
           int ncomp,
           BandMode mode,
           double constant_scale,
           int grid_version)
        { applyBandRows(out, phi, table, ncomp, mode, constant_scale, grid_version); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("table"),
        nb::arg("ncomp"),
        nb::arg("mode"),
        nb::arg("constant_scale"),
        nb::arg("grid_version"),
        "The band sweep: out(target, n) = sum_{k < nnz} a(r, k) * phi(stencil(r, k), n) "
        "+ constant_scale * c(r, n), with Overwrite writing and Add accumulating. "
        "constant_scale = 1.0 is the affine apply, 0.0 is the linear part alone (matvec mode). "
        "'out' and 'phi' must be different MultiFabs unless every row has nnz = 0 (the "
        "non-fluid pin, which reads nothing). Raises RuntimeError if grid_version or the local "
        "box count does not match the table, or if the rows index outside the MultiFabs "
        "(stencils outside phi's fab box — the ghost width is too narrow — or targets outside "
        "the valid box)."
    );
}
