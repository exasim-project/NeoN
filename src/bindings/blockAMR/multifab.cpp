// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_IndexType.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Arena.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_TagBox.H>
#include <nanobind/stl/string.h>

#include <nanobind/stl/vector.h>

#include <cstring>
#include <utility>
#include <vector>

namespace nb = nanobind;

// Arena wrapping an externally-owned pointer (e.g. from JAX/numpy).
// The first alloc() of the exact chunk size returns the external pointer;
// all subsequent allocations (staging buffers, etc.) go to a fallback arena.
class ExternalArena final : public amrex::Arena
{
    void* m_ptr;
    std::size_t m_size;
    amrex::Arena* m_fallback;
    bool m_chunk_served;
    bool m_device_accessible;
    bool m_host_accessible;

public:
    ExternalArena(
        void* p,
        std::size_t sz,
        amrex::Arena* fallback,
        bool device,
        bool host
    )
        : m_ptr(p),
          m_size(sz),
          m_fallback(fallback),
          m_chunk_served(false),
          m_device_accessible(device),
          m_host_accessible(host)
    {}

    [[nodiscard]] void* alloc(std::size_t sz) override
    {
        if (!m_chunk_served && sz == m_size)
        {
            m_chunk_served = true;
            return m_ptr;
        }
        return m_fallback->alloc(sz);
    }

    void free(void* pt) override
    {
        if (pt == m_ptr) { return; }
        m_fallback->free(pt);
    }

    [[nodiscard]] bool isDeviceAccessible() const override { return m_device_accessible; }

    [[nodiscard]] bool isHostAccessible() const override { return m_host_accessible; }

    [[nodiscard]] bool isManaged() const override
    {
        return m_device_accessible && m_host_accessible;
    }

    [[nodiscard]] bool isDevice() const override
    {
        return m_device_accessible && !m_host_accessible;
    }

    [[nodiscard]] bool isPinned() const override
    {
        return m_host_accessible && m_device_accessible;
    }
};

// Async copy from ndarray into a single FAB's valid region.
// Returns (staging_ptr, owns_staging).  Caller must call
// streamSynchronize() before freeing staging_ptr (when owns == true).
static std::pair<amrex::Real*, bool>
copyToFab_async(amrex::MultiFab& mf, amrex::MFIter& mfi, nb::ndarray<nb::ro> src)
{
    using namespace amrex;
    auto& fab = mf[mfi];
    const Box& bx = mfi.validbox();
    int ncomp = (src.ndim() == 4) ? static_cast<int>(src.shape(3)) : 1;
    int nx = bx.length(0);
    int ny = bx.length(1);
    int nz = bx.length(2);
    size_t nbytes = (size_t)bx.numPts() * ncomp * sizeof(Real);

    const Real* srcPtr = static_cast<const Real*>(src.data());
    bool srcOnDevice = (src.device_type() != nb::device::cpu::value);
    bool dstOnDevice = mf.arena()->isDeviceAccessible() && !mf.arena()->isHostAccessible();

    bool srcIsFortran = false;
    if (src.ndim() >= 2)
    {
        srcIsFortran = (src.stride(0) <= src.stride(src.ndim() - 1));
    }

    // Stage source onto the dest arena
    Real* devSrc;
    bool ownDevSrc = false;
    if (srcOnDevice == dstOnDevice)
    {
        devSrc = const_cast<Real*>(srcPtr);
    }
    else if (dstOnDevice)
    {
        devSrc = static_cast<Real*>(mf.arena()->alloc(nbytes));
        ownDevSrc = true;
        Gpu::htod_memcpy(devSrc, srcPtr, nbytes);
    }
    else
    {
        devSrc = static_cast<Real*>(The_Pinned_Arena()->alloc(nbytes));
        ownDevSrc = true;
        Gpu::dtoh_memcpy(devSrc, srcPtr, nbytes);
    }

    if (srcIsFortran)
    {
        auto* dstPtr = fab.dataPtr();
        const auto fabBox = fab.box();
        if (fabBox == bx)
        {
            if (dstOnDevice)
                Gpu::dtod_memcpy_async(dstPtr, devSrc, nbytes);
            else
                std::memcpy(dstPtr, devSrc, nbytes);
        }
        else
        {
            auto arr4 = fab.array();
            const auto lo = bx.smallEnd();
            const Real* fSrc = devSrc;
            if (dstOnDevice)
            {
                ParallelFor(
                    bx,
                    ncomp,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        int li = i - lo[0];
                        int lj = j - lo[1];
                        int lk = k - lo[2];
                        size_t fIdx = (size_t)li + (size_t)lj * nx + (size_t)lk * nx * ny
                                    + (size_t)n * nx * ny * nz;
                        arr4(i, j, k, n) = fSrc[fIdx];
                    }
                );
            }
            else
            {
                const auto lo3 = lbound(bx);
                const auto hi3 = ubound(bx);
                for (int n = 0; n < ncomp; ++n)
                    for (int k = lo3.z; k <= hi3.z; ++k)
                        for (int j = lo3.y; j <= hi3.y; ++j)
                            for (int i = lo3.x; i <= hi3.x; ++i)
                            {
                                int li = i - lo[0];
                                int lj = j - lo[1];
                                int lk = k - lo[2];
                                size_t fIdx = (size_t)li + (size_t)lj * nx + (size_t)lk * nx * ny
                                            + (size_t)n * nx * ny * nz;
                                arr4(i, j, k, n) = fSrc[fIdx];
                            }
            }
        }
    }
    else
    {
        // C-order layout: shape (nx, ny, nz, ncomp) with component as
        // the innermost (fastest-varying) dimension.
        auto arr4 = fab.array();
        const auto lo = bx.smallEnd();
        const Real* cSrc = devSrc;
        if (dstOnDevice)
        {
            ParallelFor(
                bx,
                ncomp,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    int li = i - lo[0];
                    int lj = j - lo[1];
                    int lk = k - lo[2];
                    size_t cIdx = ((size_t)li * ny * nz * ncomp)
                                + ((size_t)lj * nz * ncomp) + ((size_t)lk * ncomp) + n;
                    arr4(i, j, k, n) = cSrc[cIdx];
                }
            );
        }
        else
        {
            const auto lo3 = lbound(bx);
            const auto hi3 = ubound(bx);
            for (int n = 0; n < ncomp; ++n)
                for (int k = lo3.z; k <= hi3.z; ++k)
                    for (int j = lo3.y; j <= hi3.y; ++j)
                        for (int i = lo3.x; i <= hi3.x; ++i)
                        {
                            int li = i - lo[0];
                            int lj = j - lo[1];
                            int lk = k - lo[2];
                            size_t cIdx = ((size_t)li * ny * nz * ncomp)
                                        + ((size_t)lj * nz * ncomp) + ((size_t)lk * ncomp) + n;
                            arr4(i, j, k, n) = cSrc[cIdx];
                        }
        }
    }

    return {ownDevSrc ? devSrc : nullptr, ownDevSrc};
}

inline int deviceTypeFromArena(const amrex::MultiFab& mf)
{
    if (!mf.arena()->isDeviceAccessible()) return nb::device::cpu::value;
    // Pinned memory is host-resident (CPU-addressable) — report CPU so JAX
    // can use it even when running in CPU-only mode.
    if (mf.arena()->isHostAccessible()) return nb::device::cpu::value;
#if defined(AMREX_USE_CUDA)
    if (mf.arena()->isManaged()) return nb::device::cuda_managed::value;
    return nb::device::cuda::value;
#elif defined(AMREX_USE_HIP)
    return nb::device::rocm::value;
#elif defined(AMREX_USE_SYCL)
    return nb::device::oneapi::value;
#else
    return nb::device::cpu::value;
#endif
}

// Device tag for arrays allocated on The_Device_Arena(). On a CPU-only AMReX
// build that arena is host memory, so the exported ndarray must report CPU (a
// hardcoded cuda tag makes JAX's from_dlpack request a nonexistent cuda backend
// when running CPU-only).
#if defined(AMREX_USE_CUDA)
constexpr int kDeviceArenaDevice = nb::device::cuda::value;
#elif defined(AMREX_USE_HIP)
constexpr int kDeviceArenaDevice = nb::device::rocm::value;
#elif defined(AMREX_USE_SYCL)
constexpr int kDeviceArenaDevice = nb::device::oneapi::value;
#else
constexpr int kDeviceArenaDevice = nb::device::cpu::value;
#endif

// Python-friendly MFIter wrapper that supports __iter__/__next__
struct MFIterator
{
    amrex::MultiFab* mf;
    std::unique_ptr<amrex::MFIter> mfi;
    bool needsAdvance;

    explicit MFIterator(amrex::MultiFab& mf_) : mf(&mf_), mfi(nullptr), needsAdvance(false) {}

    amrex::MFIter& get() { return *mfi; }
};

void registerMultiFab(nb::module_& m)
{
    using namespace amrex;

    // --- BoxList ---
    nb::class_<BoxList>(m, "BoxList")
        .def(nb::init<>())
        .def(
            "push_back",
            [](BoxList& bl, const Box& bx) { bl.push_back(bx); },
            nb::arg("bx")
        )
        .def("size", &BoxList::size);

    nb::class_<BoxArray>(m, "BoxArray")
        .def(
            "__init__",
            [](BoxArray* self, const Box& bx, std::optional<IndexType> index_type)
            {
                new (self) BoxArray(bx);
                if (index_type) self->convert(*index_type);
            },
            nb::arg("bx"),
            nb::arg("index_type") = nb::none()
        )
        .def(
            "__init__",
            [](BoxArray* self, const BoxList& bl) { new (self) BoxArray(bl); },
            nb::arg("box_list")
        )
        .def(
            "__init__",
            [](BoxArray* self, const BoxArray& other) { new (self) BoxArray(other); },
            nb::arg("other")
        )
        .def(
            "max_size",
            [](BoxArray& ba, int sz) -> BoxArray& { return ba.maxSize(sz); },
            nb::arg("max_size"),
            nb::rv_policy::reference
        )
        .def("ix_type", &BoxArray::ixType)
        .def(
            "convert",
            [](BoxArray& ba, IndexType t) -> BoxArray& { return ba.convert(t); },
            nb::arg("typ"),
            nb::rv_policy::reference
        )
        .def(
            "convert",
            [](BoxArray& ba, const IntVect& t) -> BoxArray& { return ba.convert(t); },
            nb::arg("typ"),
            nb::rv_policy::reference
        )
        .def(
            "surrounding_nodes",
            [](BoxArray& ba) -> BoxArray& { return ba.surroundingNodes(); },
            nb::rv_policy::reference
        )
        .def(
            "surrounding_nodes",
            [](BoxArray& ba, int dir) -> BoxArray& { return ba.surroundingNodes(dir); },
            nb::arg("dir"),
            nb::rv_policy::reference
        )
        .def(
            "enclosed_cells",
            [](BoxArray& ba) -> BoxArray& { return ba.enclosedCells(); },
            nb::rv_policy::reference
        )
        .def(
            "enclosed_cells",
            [](BoxArray& ba, int dir) -> BoxArray& { return ba.enclosedCells(dir); },
            nb::arg("dir"),
            nb::rv_policy::reference
        );

    // Free function: create independent nodal BA from cell BA (no shared ownership)
    m.def(
        "convert_ba",
        [](const BoxArray& ba, const IntVect& typ) { return amrex::convert(ba, typ); },
        nb::arg("ba"),
        nb::arg("typ")
    );
    m.def("node_type", []() { return IntVect::TheNodeVector(); });

    nb::class_<DistributionMapping>(m, "DistributionMapping")
        .def(nb::init<const BoxArray&>(), nb::arg("ba"));

    nb::class_<MFIter>(m, "MFIter")
        .def(nb::init<const MultiFab&>(), nb::arg("mf"), nb::keep_alive<1, 2>())
        .def("is_valid", &MFIter::isValid)
        .def("valid_box", &MFIter::validbox)
        .def("_incr", &MFIter::operator++)
        .def("finalize", &MFIter::Finalize);


    nb::class_<MFIterator>(m, "MFIterator")
        .def(
            "__init__",
            [](MFIterator* self, MultiFab& mf) { new (self) MFIterator(mf); },
            nb::arg("mf"),
            nb::keep_alive<1, 2>()
        )
        .def(
            "__iter__",
            [](MFIterator& self) -> MFIterator&
            {
                self.mfi = std::make_unique<MFIter>(*self.mf);
                self.needsAdvance = false;
                return self;
            },
            nb::rv_policy::reference
        )
        .def(
            "__next__",
            [](nb::object pySelf) -> nb::object
            {
                MFIterator& self = nb::cast<MFIterator&>(pySelf);
                // Advance from previous iteration
                if (self.needsAdvance)
                {
                    ++(*self.mfi);
                }
                if (!self.mfi || !self.mfi->isValid())
                {
                    self.mfi.reset();
                    throw nb::stop_iteration();
                }
                self.needsAdvance = true;
                return pySelf;
            }
        )
        .def("valid_box", [](MFIterator& self) { return self.mfi->validbox(); })
        .def(
            "get",
            [](MFIterator& self) -> MFIter& { return self.get(); },
            nb::rv_policy::reference_internal
        );

    nb::class_<MultiFab>(m, "MultiFab")
        .def(
            "__init__",
            [](MultiFab* self,
               const BoxArray& ba,
               const DistributionMapping& dm,
               int ncomp,
               int ngrow,
               const std::string& memory)
            {
                MFInfo info;
                info.SetAllocSingleChunk(true);
                if (memory == "device") info.SetArena(The_Device_Arena());
                else if (memory == "managed")
                    info.SetArena(The_Managed_Arena());
                else if (memory == "pinned")
                    info.SetArena(The_Pinned_Arena());
                // "default" → no SetArena call, uses AMReX default
                new (self) MultiFab(ba, dm, ncomp, ngrow, info);
            },
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("ncomp"),
            nb::arg("ngrow"),
            nb::arg("memory") = "default"
        )
        .def(
            "__init__",
            [](MultiFab* self,
               const BoxArray& ba,
               const DistributionMapping& dm,
               int ncomp,
               int ngrow,
               nb::ndarray<> data)
            {
                bool onDevice = (data.device_type() != nb::device::cpu::value);
                Arena* fallback = onDevice ? The_Device_Arena() : The_Arena();
                bool devAccess = onDevice;
                bool hostAccess = !onDevice;

                size_t expected = 0;
                IntVect ng(ngrow);
                for (int i = 0; i < ba.size(); ++i)
                {
                    if (dm[i] == ParallelDescriptor::MyProc())
                    {
                        Box grown = amrex::grow(ba[i], ng);
                        expected += static_cast<size_t>(grown.numPts()) * ncomp;
                    }
                }
                size_t provided = data.size();
                if (provided != expected)
                {
                    throw std::runtime_error(
                        "Buffer size mismatch: expected "
                        + std::to_string(expected) + " elements, got "
                        + std::to_string(provided));
                }

                // ExternalArena is leaked (~64 bytes) — its lifetime exceeds
                // the MultiFab because SingleChunkArena holds a raw pointer.
                // A capsule-based cleanup can be added later.
                auto* arena = new ExternalArena(
                    data.data(), expected * sizeof(Real),
                    fallback, devAccess, hostAccess);

                MFInfo info;
                info.SetAllocSingleChunk(true);
                info.SetArena(arena);
                new (self) MultiFab(ba, dm, ncomp, ngrow, info);
            },
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("ncomp"),
            nb::arg("ngrow"),
            nb::arg("data"),
            nb::keep_alive<1, 6>()
        )
        .def_static(
            "required_buffer_size",
            [](const BoxArray& ba, const DistributionMapping& dm,
               int ncomp, int ngrow)
            {
                IntVect ng(ngrow);
                int64_t total = 0;
                for (int i = 0; i < ba.size(); ++i)
                {
                    if (dm[i] == ParallelDescriptor::MyProc())
                    {
                        Box grown = amrex::grow(ba[i], ng);
                        total += static_cast<int64_t>(grown.numPts()) * ncomp;
                    }
                }
                return total;
            },
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("ncomp"),
            nb::arg("ngrow")
        )
        .def("num_comp", &MultiFab::nComp)
        .def("n_grow", [](const MultiFab& mf) { return mf.nGrow(); })
        .def(
            "set_val",
            [](MultiFab& mf, double val) { mf.setVal(val); },
            nb::arg("val")
        )
        .def_prop_ro(
            "is_device",
            [](const MultiFab& mf)
            { return mf.arena()->isDeviceAccessible() && !mf.arena()->isHostAccessible(); }
        )
        .def_prop_ro("is_managed", [](const MultiFab& mf) { return mf.arena()->isManaged(); })
        .def_prop_ro(
            "is_host",
            [](const MultiFab& mf)
            { return mf.arena()->isHostAccessible() && !mf.arena()->isDeviceAccessible(); }
        )
        .def(
            "array",
            [](nb::object self, MFIterator& mfi)
            {
                MultiFab& mf = nb::cast<MultiFab&>(self);
                auto& fab = mf[mfi.get()];
                const Box& bx = fab.box();
                int nx = bx.length(0);
                int ny = bx.length(1);
                int nz = bx.length(2);
                int nc = mf.nComp();
                Real* ptr = fab.dataPtr();

                size_t shape[4] = {(size_t)nx, (size_t)ny, (size_t)nz, (size_t)nc};
                int devType = deviceTypeFromArena(mf);
                return nb::ndarray<nb::jax, Real, nb::ndim<4>>(
                    ptr, 4, shape, self, nullptr, nb::dtype<Real>(), devType, 0, 'F'
                );
            },
            nb::arg("mfi")
        )
        .def(
            "arrays",
            [](nb::object self)
            {
                MultiFab& mf = nb::cast<MultiFab&>(self);
                int nc = mf.nComp();
                int devType = deviceTypeFromArena(mf);
                nb::list result;
                for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
                {
                    auto& fab = mf[mfi];
                    const auto& bx = fab.box();
                    size_t shape[4] = {
                        (size_t)bx.length(0), (size_t)bx.length(1),
                        (size_t)bx.length(2), (size_t)nc
                    };
                    result.append(nb::ndarray<nb::jax, Real, nb::ndim<4>>(
                        fab.dataPtr(), 4, shape, self, nullptr,
                        nb::dtype<Real>(), devType, 0, 'F'
                    ));
                }
                return result;
            }
        )
        .def(
            "grown_arrays",
            [](nb::object self)
            {
                MultiFab& mf = nb::cast<MultiFab&>(self);
                int nc = mf.nComp();
                int devType = deviceTypeFromArena(mf);
                nb::list result;
                for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
                {
                    auto& fab = mf[mfi];
                    const auto& bx = fab.box();
                    size_t shape[4] = {
                        (size_t)bx.length(0), (size_t)bx.length(1),
                        (size_t)bx.length(2), (size_t)nc
                    };
                    result.append(nb::ndarray<nb::jax, Real, nb::ndim<4>>(
                        fab.dataPtr(), 4, shape, self, nullptr,
                        nb::dtype<Real>(), devType, 0, 'F'
                    ));
                }
                return result;
            }
        )
        .def(
            "grown_array",
            [](nb::object self, MFIterator& mfi)
            {
                MultiFab& mf = nb::cast<MultiFab&>(self);
                auto& fab = mf[mfi.get()];
                const Box& bx = fab.box();
                int nx = bx.length(0);
                int ny = bx.length(1);
                int nz = bx.length(2);
                int nc = mf.nComp();
                Real* ptr = fab.dataPtr();

                size_t shape[4] = {(size_t)nx, (size_t)ny, (size_t)nz, (size_t)nc};
                int devType = deviceTypeFromArena(mf);
                return nb::ndarray<nb::jax, Real, nb::ndim<4>>(
                    ptr, 4, shape, self, nullptr, nb::dtype<Real>(), devType, 0, 'F'
                );
            },
            nb::arg("mfi")
        )
        .def(
            "contiguous_array",
            [](nb::object self)
            {
                MultiFab& mf = nb::cast<MultiFab&>(self);
                auto* ptr = mf.singleChunkPtr();
                if (!ptr)
                    throw std::runtime_error(
                        "MultiFab not allocated with single-chunk mode");
                size_t total = mf.singleChunkSize() / sizeof(Real);
                size_t shape[1] = {total};
                int devType = deviceTypeFromArena(mf);
                return nb::ndarray<nb::jax, Real, nb::ndim<1>>(
                    ptr, 1, shape, self, nullptr,
                    nb::dtype<Real>(), devType, 0, 'F'
                );
            }
        )
        .def(
            "copy_from_flat",
            [](MultiFab& mf, nb::ndarray<nb::ro> src)
            {
                auto* dst = mf.singleChunkPtr();
                if (!dst)
                    throw std::runtime_error(
                        "MultiFab not allocated with single-chunk mode");
                size_t nbytes = mf.singleChunkSize();
                size_t src_nbytes = src.size() * src.itemsize();
                if (src_nbytes != nbytes)
                    throw std::runtime_error(
                        "copy_from_flat: size mismatch — src has "
                        + std::to_string(src_nbytes) + " bytes, MultiFab has "
                        + std::to_string(nbytes));
                const void* src_ptr = src.data();
                bool srcOnDevice =
                    (src.device_type() != nb::device::cpu::value);
                if (srcOnDevice)
                    Gpu::dtod_memcpy(dst, src_ptr, nbytes);
                else
                    Gpu::htod_memcpy(dst, src_ptr, nbytes);
            },
            nb::arg("flat_array"),
            "Copy flat array directly into MultiFab contiguous storage. "
            "Single memcpy — no per-box loop. Array must have same byte size "
            "as singleChunkSize()."
        )
        .def(
            "fab_metadata",
            [](MultiFab& mf)
            {
                nb::list result;
                size_t offset = 0;
                int nc = mf.nComp();
                for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
                {
                    const auto& bx = mf[mfi].box();
                    int nx = bx.length(0);
                    int ny = bx.length(1);
                    int nz = bx.length(2);
                    result.append(nb::make_tuple(
                        (int64_t)offset, nx, ny, nz, nc));
                    offset += (size_t)nx * ny * nz * nc;
                }
                return result;
            }
        )
        .def(
            "tile_table",
            [](MultiFab& mf, int bf, int requested_padded)
            {
                if (!mf.singleChunkPtr())
                    throw std::runtime_error(
                        "MultiFab not allocated with single-chunk mode");

                int ng = mf.nGrow();
                int nc = mf.nComp();

                // First pass: count tiles
                size_t n_tiles = 0;
                size_t box_offset = 0;
                for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
                {
                    const auto& bx = mf[mfi].box();
                    int Nx = bx.length(0);
                    int Ny = bx.length(1);
                    int Nz = bx.length(2);
                    int vNx = Nx - 2 * ng;
                    int vNy = Ny - 2 * ng;
                    int vNz = Nz - 2 * ng;
                    n_tiles += (size_t)(vNx / bf) * (vNy / bf) * (vNz / bf);
                    box_offset += (size_t)Nx * Ny * Nz * nc;
                }

                // Pad: use requested size or next power of 2
                size_t n_padded;
                if (requested_padded > 0 && (size_t)requested_padded >= n_tiles)
                    n_padded = (size_t)requested_padded;
                else {
                    n_padded = 1;
                    while (n_padded < n_tiles)
                        n_padded <<= 1;
                }

                // Allocate flat arrays
                auto* offsets  = new int64_t[n_padded];
                auto* stride_x = new int64_t[n_padded];
                auto* stride_y = new int64_t[n_padded];
                auto* stride_z = new int64_t[n_padded];
                auto* stride_c = new int64_t[n_padded];
                auto* box_ids  = new int64_t[n_padded];
                auto* tile_is  = new int64_t[n_padded];
                auto* tile_js  = new int64_t[n_padded];
                auto* tile_ks  = new int64_t[n_padded];

                // Second pass: fill tile descriptors
                size_t t = 0;
                box_offset = 0;
                int box_id = 0;
                for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
                {
                    const auto& bx = mf[mfi].box();
                    int Nx = bx.length(0);
                    int Ny = bx.length(1);
                    int Nz = bx.length(2);
                    int vNx = Nx - 2 * ng;
                    int vNy = Ny - 2 * ng;
                    int vNz = Nz - 2 * ng;

                    // Fortran-order strides for this box
                    int64_t sx = 1;
                    int64_t sy = Nx;
                    int64_t sz = (int64_t)Nx * Ny;
                    int64_t sc = (int64_t)Nx * Ny * Nz;

                    for (int ti = 0; ti < vNx / bf; ++ti)
                    {
                        for (int tj = 0; tj < vNy / bf; ++tj)
                        {
                            for (int tk = 0; tk < vNz / bf; ++tk)
                            {
                                // Corner of this tile's ghost origin.
                                // The kernel adds ng internally when iterating,
                                // so the offset points to (ti*bf, tj*bf, tk*bf)
                                // in the box, not (ng+ti*bf, ...).
                                int ci = ti * bf;
                                int cj = tj * bf;
                                int ck = tk * bf;
                                int64_t tile_off = (int64_t)box_offset
                                                 + ci * sx + cj * sy + ck * sz;

                                offsets[t]  = tile_off;
                                stride_x[t] = sx;
                                stride_y[t] = sy;
                                stride_z[t] = sz;
                                stride_c[t] = sc;
                                box_ids[t]  = box_id;
                                tile_is[t]  = ti;
                                tile_js[t]  = tj;
                                tile_ks[t]  = tk;
                                ++t;
                            }
                        }
                    }
                    box_offset += (size_t)Nx * Ny * Nz * nc;
                    ++box_id;
                }

                // Pad with copies of tile 0
                for (size_t p = t; p < n_padded; ++p)
                {
                    offsets[p]  = offsets[0];
                    stride_x[p] = stride_x[0];
                    stride_y[p] = stride_y[0];
                    stride_z[p] = stride_z[0];
                    stride_c[p] = stride_c[0];
                    box_ids[p]  = 0;
                    tile_is[p]  = 0;
                    tile_js[p]  = 0;
                    tile_ks[p]  = 0;
                }

                // Copy to device and return as JAX arrays
                auto make_array = [&](int64_t* host_ptr) {
                    size_t nbytes = n_padded * sizeof(int64_t);
                    auto* dev_ptr = static_cast<int64_t*>(
                        The_Device_Arena()->alloc(nbytes));
                    Gpu::htod_memcpy(dev_ptr, host_ptr, nbytes);
                    delete[] host_ptr;
                    auto owner = nb::capsule(dev_ptr, [](void* p) noexcept {
                        The_Device_Arena()->free(p);
                    });
                    size_t shape[1] = {n_padded};
                    return nb::ndarray<nb::jax, int64_t, nb::ndim<1>>(
                        dev_ptr, 1, shape, owner, nullptr,
                        nb::dtype<int64_t>(), kDeviceArenaDevice, 0);
                };

                nb::dict result;
                result["offset"]   = make_array(offsets);
                result["stride_x"] = make_array(stride_x);
                result["stride_y"] = make_array(stride_y);
                result["stride_z"] = make_array(stride_z);
                result["stride_c"] = make_array(stride_c);
                result["box_id"]   = make_array(box_ids);
                result["tile_i"]   = make_array(tile_is);
                result["tile_j"]   = make_array(tile_js);
                result["tile_k"]   = make_array(tile_ks);
                result["n_tiles"]  = nb::int_(n_tiles);
                result["n_padded"] = nb::int_(n_padded);
                result["bf"]       = nb::int_(bf);
                result["ng"]       = nb::int_(ng);
                return result;
            },
            nb::arg("bf") = 4, nb::arg("n_padded") = 0
        )
        .def(
            "packed_tiles",
            [](MultiFab& mf, int bf, int requested_padded)
            {
                if (!mf.singleChunkPtr())
                    throw std::runtime_error(
                        "MultiFab not allocated with single-chunk mode");

                int ng = mf.nGrow();
                int nc = mf.nComp();
                constexpr int FIELDS_PER_TILE = 5;

                // Single pass: count tiles and collect per-box info
                size_t n_tiles = 0;
                size_t box_offset = 0;

                struct BoxInfo {
                    size_t offset;
                    int Nx, Ny, Nz;
                    int vNx, vNy, vNz;
                    int box_id;
                };
                std::vector<BoxInfo> boxes;

                int box_id = 0;
                for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
                {
                    const auto& bx = mf[mfi].box();
                    int Nx = bx.length(0);
                    int Ny = bx.length(1);
                    int Nz = bx.length(2);
                    int vNx = Nx - 2 * ng;
                    int vNy = Ny - 2 * ng;
                    int vNz = Nz - 2 * ng;

                    if (box_offset > (size_t)INT32_MAX)
                        throw std::runtime_error(
                            "packed_tiles: buffer offset exceeds int32 range");

                    boxes.push_back({box_offset, Nx, Ny, Nz, vNx, vNy, vNz, box_id});
                    n_tiles += (size_t)(vNx / bf) * (vNy / bf) * (vNz / bf);
                    box_offset += (size_t)Nx * Ny * Nz * nc;
                    ++box_id;
                }

                // Pad to next power of 2
                size_t n_padded;
                if (requested_padded > 0 && (size_t)requested_padded >= n_tiles)
                    n_padded = (size_t)requested_padded;
                else {
                    n_padded = 1;
                    while (n_padded < n_tiles)
                        n_padded <<= 1;
                }

                // Allocate single packed host array
                auto* packed = new int32_t[n_padded * FIELDS_PER_TILE];

                // Fill tile descriptors
                size_t t = 0;
                for (const auto& bi : boxes)
                {
                    int32_t sx = 1;
                    int32_t sy = bi.Nx;
                    int32_t sz = bi.Nx * bi.Ny;

                    for (int ti = 0; ti < bi.vNx / bf; ++ti)
                    {
                        for (int tj = 0; tj < bi.vNy / bf; ++tj)
                        {
                            for (int tk = 0; tk < bi.vNz / bf; ++tk)
                            {
                                int ci = ng + ti * bf;
                                int cj = ng + tj * bf;
                                int ck = ng + tk * bf;
                                int32_t offset = (int32_t)bi.offset
                                               + ci * sx + cj * sy + ck * sz;

                                size_t base = t * FIELDS_PER_TILE;
                                packed[base + 0] = offset;
                                packed[base + 1] = sx;
                                packed[base + 2] = sy;
                                packed[base + 3] = sz;
                                packed[base + 4] = bi.box_id;
                                ++t;
                            }
                        }
                    }
                }

                // Pad with copies of tile 0
                for (size_t p = t; p < n_padded; ++p)
                {
                    size_t dst = p * FIELDS_PER_TILE;
                    for (int f = 0; f < FIELDS_PER_TILE; ++f)
                        packed[dst + f] = packed[f];
                }

                // Single htod_memcpy to device
                size_t nbytes = n_padded * FIELDS_PER_TILE * sizeof(int32_t);
                auto* dev_ptr = static_cast<int32_t*>(
                    The_Device_Arena()->alloc(nbytes));
                Gpu::htod_memcpy(dev_ptr, packed, nbytes);
                delete[] packed;

                auto owner = nb::capsule(dev_ptr, [](void* p) noexcept {
                    The_Device_Arena()->free(p);
                });
                size_t shape[1] = {n_padded * FIELDS_PER_TILE};
                auto tiles_arr = nb::ndarray<nb::jax, int32_t, nb::ndim<1>>(
                    dev_ptr, 1, shape, owner, nullptr,
                    nb::dtype<int32_t>(), kDeviceArenaDevice, 0);

                nb::dict result;
                result["tiles"]    = tiles_arr;
                result["n_tiles"]  = nb::int_(n_tiles);
                result["n_padded"] = nb::int_(n_padded);
                result["bf"]       = nb::int_(bf);
                result["ng"]       = nb::int_(ng);
                return result;
            },
            nb::arg("bf") = 8, nb::arg("n_padded") = 0,
            "Build packed tile metadata: [offset, sx, sy, sz, box_id] per tile."
        )
        .def(
            "face_tile_table",
            [](MultiFab& cell_mf,
               const MultiFab& fx_mf,
               const MultiFab& fy_mf,
               const MultiFab& fz_mf,
               int bf,
               int requested_padded)
            {
                if (!cell_mf.singleChunkPtr())
                    throw std::runtime_error(
                        "Cell MultiFab not allocated with single-chunk mode");

                int ng = cell_mf.nGrow();
                int nc = cell_mf.nComp();

                // Pass 1: count tiles
                size_t n_tiles = 0;
                for (amrex::MFIter mfi(cell_mf); mfi.isValid(); ++mfi)
                {
                    const auto& bx = cell_mf[mfi].box();
                    int vNx = bx.length(0) - 2 * ng;
                    int vNy = bx.length(1) - 2 * ng;
                    int vNz = bx.length(2) - 2 * ng;
                    n_tiles += (size_t)(vNx / bf) * (vNy / bf) * (vNz / bf);
                }

                size_t n_padded;
                if (requested_padded > 0 && (size_t)requested_padded >= n_tiles)
                    n_padded = (size_t)requested_padded;
                else {
                    n_padded = 1;
                    while (n_padded < n_tiles)
                        n_padded <<= 1;
                }

                // 24 arrays: 3 dirs × (off, sx, sy, sz, maxI, maxJ, maxK) + tile_ci/cj/ck
                auto* fx_off  = new int32_t[n_padded];
                auto* fx_sx   = new int32_t[n_padded];
                auto* fx_sy   = new int32_t[n_padded];
                auto* fx_sz   = new int32_t[n_padded];
                auto* fx_maxI = new int32_t[n_padded];
                auto* fx_maxJ = new int32_t[n_padded];
                auto* fx_maxK = new int32_t[n_padded];

                auto* fy_off  = new int32_t[n_padded];
                auto* fy_sx   = new int32_t[n_padded];
                auto* fy_sy   = new int32_t[n_padded];
                auto* fy_sz   = new int32_t[n_padded];
                auto* fy_maxI = new int32_t[n_padded];
                auto* fy_maxJ = new int32_t[n_padded];
                auto* fy_maxK = new int32_t[n_padded];

                auto* fz_off  = new int32_t[n_padded];
                auto* fz_sx   = new int32_t[n_padded];
                auto* fz_sy   = new int32_t[n_padded];
                auto* fz_sz   = new int32_t[n_padded];
                auto* fz_maxI = new int32_t[n_padded];
                auto* fz_maxJ = new int32_t[n_padded];
                auto* fz_maxK = new int32_t[n_padded];

                auto* tile_ci = new int32_t[n_padded];
                auto* tile_cj = new int32_t[n_padded];
                auto* tile_ck = new int32_t[n_padded];

                // Pass 2: fill — iterate cell MFIter, use index for face access
                size_t t = 0;
                size_t fx_box_off = 0, fy_box_off = 0, fz_box_off = 0;
                int box_idx = 0;

                for (amrex::MFIter mfi(cell_mf); mfi.isValid(); ++mfi)
                {
                    const auto& cell_bx = cell_mf[mfi].box();
                    int Nx = cell_bx.length(0);
                    int Ny = cell_bx.length(1);
                    int Nz = cell_bx.length(2);
                    int vNx = Nx - 2 * ng;
                    int vNy = Ny - 2 * ng;
                    int vNz = Nz - 2 * ng;

                    // Face FAB dimensions — access by index (same ordering)
                    const auto& fx_bx = fx_mf.boxArray()[box_idx];
                    int fxNx = fx_bx.length(0), fxNy = fx_bx.length(1), fxNz = fx_bx.length(2);
                    // Add ngrow for face
                    int fx_ng = fx_mf.nGrow();
                    fxNx += 2 * fx_ng; fxNy += 2 * fx_ng; fxNz += 2 * fx_ng;

                    const auto& fy_bx = fy_mf.boxArray()[box_idx];
                    int fyNx = fy_bx.length(0), fyNy = fy_bx.length(1), fyNz = fy_bx.length(2);
                    int fy_ng = fy_mf.nGrow();
                    fyNx += 2 * fy_ng; fyNy += 2 * fy_ng; fyNz += 2 * fy_ng;

                    const auto& fz_bx = fz_mf.boxArray()[box_idx];
                    int fzNx = fz_bx.length(0), fzNy = fz_bx.length(1), fzNz = fz_bx.length(2);
                    int fz_ng = fz_mf.nGrow();
                    fzNx += 2 * fz_ng; fzNy += 2 * fz_ng; fzNz += 2 * fz_ng;

                    for (int ti = 0; ti < vNx / bf; ++ti)
                    {
                        for (int tj = 0; tj < vNy / bf; ++tj)
                        {
                            for (int tk = 0; tk < vNz / bf; ++tk)
                            {
                                int ci = ti * bf;
                                int cj = tj * bf;
                                int ck = tk * bf;

                                // x-face
                                fx_off[t]  = (int32_t)fx_box_off;
                                fx_sx[t]   = 1;
                                fx_sy[t]   = fxNx;
                                fx_sz[t]   = fxNx * fxNy;
                                fx_maxI[t] = fxNx;
                                fx_maxJ[t] = fxNy;
                                fx_maxK[t] = fxNz;

                                // y-face
                                fy_off[t]  = (int32_t)fy_box_off;
                                fy_sx[t]   = 1;
                                fy_sy[t]   = fyNx;
                                fy_sz[t]   = fyNx * fyNy;
                                fy_maxI[t] = fyNx;
                                fy_maxJ[t] = fyNy;
                                fy_maxK[t] = fyNz;

                                // z-face
                                fz_off[t]  = (int32_t)fz_box_off;
                                fz_sx[t]   = 1;
                                fz_sy[t]   = fzNx;
                                fz_sz[t]   = fzNx * fzNy;
                                fz_maxI[t] = fzNx;
                                fz_maxJ[t] = fzNy;
                                fz_maxK[t] = fzNz;

                                // Tile corner
                                tile_ci[t] = ci;
                                tile_cj[t] = cj;
                                tile_ck[t] = ck;

                                ++t;
                            }
                        }
                    }

                    fx_box_off += (size_t)fxNx * fxNy * fxNz * fx_mf.nComp();
                    fy_box_off += (size_t)fyNx * fyNy * fyNz * fy_mf.nComp();
                    fz_box_off += (size_t)fzNx * fzNy * fzNz * fz_mf.nComp();
                    ++box_idx;
                }

                // Pad with copies of tile 0
                for (size_t p = t; p < n_padded; ++p)
                {
                    fx_off[p] = fx_off[0]; fx_sx[p] = fx_sx[0];
                    fx_sy[p] = fx_sy[0]; fx_sz[p] = fx_sz[0];
                    fx_maxI[p] = fx_maxI[0]; fx_maxJ[p] = fx_maxJ[0]; fx_maxK[p] = fx_maxK[0];
                    fy_off[p] = fy_off[0]; fy_sx[p] = fy_sx[0];
                    fy_sy[p] = fy_sy[0]; fy_sz[p] = fy_sz[0];
                    fy_maxI[p] = fy_maxI[0]; fy_maxJ[p] = fy_maxJ[0]; fy_maxK[p] = fy_maxK[0];
                    fz_off[p] = fz_off[0]; fz_sx[p] = fz_sx[0];
                    fz_sy[p] = fz_sy[0]; fz_sz[p] = fz_sz[0];
                    fz_maxI[p] = fz_maxI[0]; fz_maxJ[p] = fz_maxJ[0]; fz_maxK[p] = fz_maxK[0];
                    tile_ci[p] = 0; tile_cj[p] = 0; tile_ck[p] = 0;
                }

                auto make_array = [&](int32_t* host_ptr) {
                    size_t nbytes = n_padded * sizeof(int32_t);
                    auto* dev_ptr = static_cast<int32_t*>(
                        The_Device_Arena()->alloc(nbytes));
                    Gpu::htod_memcpy(dev_ptr, host_ptr, nbytes);
                    delete[] host_ptr;
                    auto owner = nb::capsule(dev_ptr, [](void* p) noexcept {
                        The_Device_Arena()->free(p);
                    });
                    size_t shape[1] = {n_padded};
                    return nb::ndarray<nb::jax, int32_t, nb::ndim<1>>(
                        dev_ptr, 1, shape, owner, nullptr,
                        nb::dtype<int32_t>(), kDeviceArenaDevice, 0);
                };

                nb::dict result;
                result["fx_off"]  = make_array(fx_off);
                result["fx_sx"]   = make_array(fx_sx);
                result["fx_sy"]   = make_array(fx_sy);
                result["fx_sz"]   = make_array(fx_sz);
                result["fx_maxI"] = make_array(fx_maxI);
                result["fx_maxJ"] = make_array(fx_maxJ);
                result["fx_maxK"] = make_array(fx_maxK);
                result["fy_off"]  = make_array(fy_off);
                result["fy_sx"]   = make_array(fy_sx);
                result["fy_sy"]   = make_array(fy_sy);
                result["fy_sz"]   = make_array(fy_sz);
                result["fy_maxI"] = make_array(fy_maxI);
                result["fy_maxJ"] = make_array(fy_maxJ);
                result["fy_maxK"] = make_array(fy_maxK);
                result["fz_off"]  = make_array(fz_off);
                result["fz_sx"]   = make_array(fz_sx);
                result["fz_sy"]   = make_array(fz_sy);
                result["fz_sz"]   = make_array(fz_sz);
                result["fz_maxI"] = make_array(fz_maxI);
                result["fz_maxJ"] = make_array(fz_maxJ);
                result["fz_maxK"] = make_array(fz_maxK);
                result["tile_ci"] = make_array(tile_ci);
                result["tile_cj"] = make_array(tile_cj);
                result["tile_ck"] = make_array(tile_ck);
                result["n_tiles"]  = nb::int_(n_tiles);
                result["n_padded"] = nb::int_(n_padded);
                result["bf"]       = nb::int_(bf);
                result["ng"]       = nb::int_(ng);
                return result;
            },
            nb::arg("fx"), nb::arg("fy"), nb::arg("fz"), nb::arg("bf") = 4,
            nb::arg("n_padded") = 0,
            "Build face tile table matching cell tile_table layout."
        )
        .def(
            "host_array",
            [](nb::object self, MFIterator& mfi)
            {
                MultiFab& mf = nb::cast<MultiFab&>(self);
                if (!mf.arena()->isHostAccessible())
                {
                    throw std::runtime_error("MultiFab is on device-only memory. "
                                             "Use copy_to_host() or managed memory.");
                }
                auto& fab = mf[mfi.get()];
                const Box& bx = mfi.get().validbox();
                auto lo = bx.smallEnd();
                int nx = bx.length(0);
                int ny = bx.length(1);
                int nz = bx.length(2);
                int nc = mf.nComp();

                const Box& fabbox = fab.box();
                int fnx = fabbox.length(0);
                int fny = fabbox.length(1);
                int fnz = fabbox.length(2);
                int ngx = lo[0] - fabbox.smallEnd(0);
                int ngy = lo[1] - fabbox.smallEnd(1);
                int ngz = lo[2] - fabbox.smallEnd(2);
                Real* validPtr = fab.dataPtr() + ngx + ngy * fnx + ngz * fnx * fny;

                size_t shape[4] = {(size_t)nx, (size_t)ny, (size_t)nz, (size_t)nc};
                int64_t strides[4] = {
                    1, (int64_t)fnx, (int64_t)(fnx * fny), (int64_t)(fnx * fny * fnz)
                };
                return nb::ndarray<nb::numpy, Real, nb::ndim<4>>(validPtr, 4, shape, self, strides);
            },
            nb::arg("mfi")
        )
        .def(
            "host_grown_array",
            [](nb::object self, MFIterator& mfi)
            {
                MultiFab& mf = nb::cast<MultiFab&>(self);
                if (!mf.arena()->isHostAccessible())
                {
                    throw std::runtime_error("MultiFab is on device-only memory. "
                                             "Use copy_to_host() or managed memory.");
                }
                auto& fab = mf[mfi.get()];
                const Box& bx = fab.box();
                int nx = bx.length(0);
                int ny = bx.length(1);
                int nz = bx.length(2);
                int nc = mf.nComp();
                Real* ptr = fab.dataPtr();

                size_t shape[4] = {(size_t)nx, (size_t)ny, (size_t)nz, (size_t)nc};
                int64_t strides[4] = {1, (int64_t)nx, (int64_t)(nx * ny), (int64_t)(nx * ny * nz)};
                return nb::ndarray<nb::numpy, Real, nb::ndim<4>>(ptr, 4, shape, self, strides);
            },
            nb::arg("mfi")
        )
        .def(
            "copy_to_host",
            [](MultiFab& mf, MFIterator& mfi)
            {
                auto& fab = mf[mfi.get()];
                const Box& bx = mfi.get().validbox();
                int nx = bx.length(0);
                int ny = bx.length(1);
                int nz = bx.length(2);
                int nc = mf.nComp();
                size_t nelems = (size_t)nx * ny * nz * nc;

                // Copy valid region to a contiguous host FAB
                FArrayBox hostFab(bx, nc, The_Pinned_Arena());
                bool srcOnDevice =
                    mf.arena()->isDeviceAccessible() && !mf.arena()->isHostAccessible();
                if (srcOnDevice)
                    hostFab.template copy<RunOn::Device>(
                        fab, bx, SrcComp {0}, DestComp {0}, NumComps {nc}
                    );
                else
                    hostFab.template copy<RunOn::Host>(
                        fab, bx, SrcComp {0}, DestComp {0}, NumComps {nc}
                    );
                amrex::Gpu::streamSynchronize();

                Real* hostBuf = new Real[nelems];
                auto owner =
                    nb::capsule(hostBuf, [](void* p) noexcept { delete[] static_cast<Real*>(p); });
                std::memcpy(hostBuf, hostFab.dataPtr(), nelems * sizeof(Real));

                size_t shape[4] = {(size_t)nx, (size_t)ny, (size_t)nz, (size_t)nc};
                return nb::ndarray<nb::numpy, Real, nb::ndim<4>>(
                    hostBuf,
                    4,
                    shape,
                    owner,
                    nullptr,
                    nb::dtype<Real>(),
                    nb::device::cpu::value,
                    0,
                    'F'
                );
            },
            nb::arg("mfi")
        )
        .def(
            "copy_grown_to_host",
            [](MultiFab& mf, MFIterator& mfi)
            {
                auto& fab = mf[mfi.get()];
                const Box& bx = fab.box();  // grown box (includes ghosts)
                int nx = bx.length(0);
                int ny = bx.length(1);
                int nz = bx.length(2);
                int nc = mf.nComp();
                size_t nelems = (size_t)nx * ny * nz * nc;

                FArrayBox hostFab(bx, nc, The_Pinned_Arena());
                bool srcOnDevice =
                    mf.arena()->isDeviceAccessible() && !mf.arena()->isHostAccessible();
                if (srcOnDevice)
                    hostFab.template copy<RunOn::Device>(
                        fab, bx, SrcComp {0}, DestComp {0}, NumComps {nc}
                    );
                else
                    hostFab.template copy<RunOn::Host>(
                        fab, bx, SrcComp {0}, DestComp {0}, NumComps {nc}
                    );
                amrex::Gpu::streamSynchronize();

                Real* hostBuf = new Real[nelems];
                auto owner =
                    nb::capsule(hostBuf, [](void* p) noexcept { delete[] static_cast<Real*>(p); });
                std::memcpy(hostBuf, hostFab.dataPtr(), nelems * sizeof(Real));

                size_t shape[4] = {(size_t)nx, (size_t)ny, (size_t)nz, (size_t)nc};
                return nb::ndarray<nb::numpy, Real, nb::ndim<4>>(
                    hostBuf,
                    4,
                    shape,
                    owner,
                    nullptr,
                    nb::dtype<Real>(),
                    nb::device::cpu::value,
                    0,
                    'F'
                );
            },
            nb::arg("mfi")
        )
        .def(
            "copy_from",
            [](MultiFab& mf, MFIterator& mfi, nb::ndarray<nb::ro> src)
            {
                auto [ptr, owns] = copyToFab_async(mf, mfi.get(), src);
                bool dstOnDevice =
                    mf.arena()->isDeviceAccessible() && !mf.arena()->isHostAccessible();
                if (dstOnDevice)
                    amrex::Gpu::streamSynchronize();
                if (owns)
                {
                    if (dstOnDevice) mf.arena()->free(ptr);
                    else
                        The_Pinned_Arena()->free(ptr);
                }
            },
            nb::arg("mfi"),
            nb::arg("src")
        )
        .def(
            "copy_grown_from",
            [](MultiFab& mf, MFIterator& mfi, nb::ndarray<nb::ro> src)
            {
                using namespace amrex;
                auto& fab = mf[mfi.get()];
                const Box& bx = fab.box();  // grown box
                int nc = mf.nComp();
                int nx = bx.length(0);
                int ny = bx.length(1);
                int nz = bx.length(2);
                size_t nbytes = (size_t)bx.numPts() * nc * sizeof(Real);

                const Real* srcPtr = static_cast<const Real*>(src.data());
                bool srcOnDevice = (src.device_type() != nb::device::cpu::value);
                bool dstOnDevice =
                    mf.arena()->isDeviceAccessible() && !mf.arena()->isHostAccessible();

                bool srcIsFortran = false;
                if (src.ndim() >= 2)
                    srcIsFortran = (src.stride(0) <= src.stride(src.ndim() - 1));

                if (srcIsFortran)
                {
                    // Same layout as AMReX FAB — direct memcpy
                    if (srcOnDevice && dstOnDevice)
                        Gpu::dtod_memcpy_async(fab.dataPtr(), srcPtr, nbytes);
                    else if (dstOnDevice)
                        Gpu::htod_memcpy_async(fab.dataPtr(), srcPtr, nbytes);
                    else if (srcOnDevice)
                        Gpu::dtoh_memcpy(fab.dataPtr(), srcPtr, nbytes);
                    else
                        std::memcpy(fab.dataPtr(), srcPtr, nbytes);
                }
                else
                {
                    // C-order: stage to correct arena and transpose
                    Real* devSrc;
                    bool ownDevSrc = false;
                    if (srcOnDevice == dstOnDevice)
                    {
                        devSrc = const_cast<Real*>(srcPtr);
                    }
                    else if (dstOnDevice)
                    {
                        devSrc = static_cast<Real*>(mf.arena()->alloc(nbytes));
                        ownDevSrc = true;
                        Gpu::htod_memcpy(devSrc, srcPtr, nbytes);
                    }
                    else
                    {
                        devSrc = static_cast<Real*>(The_Pinned_Arena()->alloc(nbytes));
                        ownDevSrc = true;
                        Gpu::dtoh_memcpy(devSrc, srcPtr, nbytes);
                    }

                    auto arr4 = fab.array();
                    const auto lo = bx.smallEnd();
                    const Real* cSrc = devSrc;
                    if (dstOnDevice)
                    {
                        ParallelFor(
                            bx,
                            nc,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                int li = i - lo[0];
                                int lj = j - lo[1];
                                int lk = k - lo[2];
                                size_t cIdx = ((size_t)li * ny * nz * nc)
                                            + ((size_t)lj * nz * nc)
                                            + ((size_t)lk * nc) + n;
                                arr4(i, j, k, n) = cSrc[cIdx];
                            });
                    }
                    else
                    {
                        const auto lo3 = lbound(bx);
                        const auto hi3 = ubound(bx);
                        for (int n = 0; n < nc; ++n)
                            for (int k = lo3.z; k <= hi3.z; ++k)
                                for (int j = lo3.y; j <= hi3.y; ++j)
                                    for (int i = lo3.x; i <= hi3.x; ++i)
                                    {
                                        int li = i - lo[0];
                                        int lj = j - lo[1];
                                        int lk = k - lo[2];
                                        size_t cIdx = ((size_t)li * ny * nz * nc)
                                                    + ((size_t)lj * nz * nc)
                                                    + ((size_t)lk * nc) + n;
                                        arr4(i, j, k, n) = cSrc[cIdx];
                                    }
                    }

                    if (dstOnDevice || srcOnDevice)
                        Gpu::streamSynchronize();
                    if (ownDevSrc)
                    {
                        if (dstOnDevice)
                            mf.arena()->free(devSrc);
                        else
                            The_Pinned_Arena()->free(devSrc);
                    }
                    return;
                }

                if (dstOnDevice || srcOnDevice)
                    Gpu::streamSynchronize();
            },
            nb::arg("mfi"),
            nb::arg("src")
        )
        .def(
            "copy_arrays",
            [](MultiFab& mf, nb::list arrays)
            {
                using StagingEntry = std::pair<amrex::Real*, bool>;
                std::vector<StagingEntry> staging;
                bool dstOnDevice =
                    mf.arena()->isDeviceAccessible() && !mf.arena()->isHostAccessible();

                int idx = 0;
                for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi, ++idx)
                {
                    auto src = nb::cast<nb::ndarray<nb::ro>>(arrays[idx]);
                    auto [ptr, owns] = copyToFab_async(mf, mfi, src);
                    if (owns) staging.emplace_back(ptr, dstOnDevice);
                }

                if (dstOnDevice)
                    amrex::Gpu::streamSynchronize();

                for (auto& [ptr, onDev] : staging)
                {
                    if (onDev) mf.arena()->free(ptr);
                    else
                        The_Pinned_Arena()->free(ptr);
                }
            },
            nb::arg("arrays")
        )
        .def(
            "copy_grown_arrays",
            [](MultiFab& mf, nb::list arrays)
            {
                using namespace amrex;
                bool dstOnDevice =
                    mf.arena()->isDeviceAccessible() && !mf.arena()->isHostAccessible();

                int idx = 0;
                for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi, ++idx)
                {
                    auto src = nb::cast<nb::ndarray<nb::ro>>(arrays[idx]);
                    auto& fab = mf[mfi];
                    const Box& bx = fab.box();  // grown box (includes ghosts)
                    int nc = mf.nComp();
                    size_t nbytes = (size_t)bx.numPts() * nc * sizeof(Real);

                    const Real* srcPtr = static_cast<const Real*>(src.data());
                    bool srcOnDevice = (src.device_type() != nb::device::cpu::value);

                    bool srcIsFortran = false;
                    if (src.ndim() >= 2)
                        srcIsFortran = (src.stride(0) <= src.stride(src.ndim() - 1));

                    if (srcIsFortran)
                    {
                        if (srcOnDevice && dstOnDevice)
                            Gpu::dtod_memcpy_async(fab.dataPtr(), srcPtr, nbytes);
                        else if (dstOnDevice)
                            Gpu::htod_memcpy_async(fab.dataPtr(), srcPtr, nbytes);
                        else if (srcOnDevice)
                            Gpu::dtoh_memcpy_async(fab.dataPtr(), srcPtr, nbytes);
                        else
                            std::memcpy(fab.dataPtr(), srcPtr, nbytes);
                    }
                    else
                    {
                        // C-order: stage + transpose
                        int nx = bx.length(0), ny = bx.length(1), nz = bx.length(2);
                        Real* devSrc;
                        bool ownDevSrc = false;
                        if (srcOnDevice == dstOnDevice)
                        {
                            devSrc = const_cast<Real*>(srcPtr);
                        }
                        else if (dstOnDevice)
                        {
                            devSrc = static_cast<Real*>(mf.arena()->alloc(nbytes));
                            ownDevSrc = true;
                            Gpu::htod_memcpy(devSrc, srcPtr, nbytes);
                        }
                        else
                        {
                            devSrc = static_cast<Real*>(The_Pinned_Arena()->alloc(nbytes));
                            ownDevSrc = true;
                            Gpu::dtoh_memcpy(devSrc, srcPtr, nbytes);
                        }

                        auto arr4 = fab.array();
                        const auto lo = bx.smallEnd();
                        const Real* cSrc = devSrc;
                        if (dstOnDevice)
                        {
                            ParallelFor(
                                bx, nc,
                                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                                {
                                    int li = i - lo[0], lj = j - lo[1], lk = k - lo[2];
                                    size_t cIdx = ((size_t)li * ny * nz * nc)
                                                + ((size_t)lj * nz * nc)
                                                + ((size_t)lk * nc) + n;
                                    arr4(i, j, k, n) = cSrc[cIdx];
                                });
                        }
                        else
                        {
                            const auto lo3 = lbound(bx);
                            const auto hi3 = ubound(bx);
                            for (int n = 0; n < nc; ++n)
                                for (int k = lo3.z; k <= hi3.z; ++k)
                                    for (int j = lo3.y; j <= hi3.y; ++j)
                                        for (int i = lo3.x; i <= hi3.x; ++i)
                                        {
                                            int li = i - lo[0], lj = j - lo[1], lk = k - lo[2];
                                            size_t cIdx = ((size_t)li * ny * nz * nc)
                                                        + ((size_t)lj * nz * nc)
                                                        + ((size_t)lk * nc) + n;
                                            arr4(i, j, k, n) = cSrc[cIdx];
                                        }
                        }

                        if (ownDevSrc)
                        {
                            if (dstOnDevice || srcOnDevice)
                                Gpu::streamSynchronize();
                            if (dstOnDevice) mf.arena()->free(devSrc);
                            else The_Pinned_Arena()->free(devSrc);
                        }
                    }
                }

                if (dstOnDevice)
                    amrex::Gpu::streamSynchronize();
            },
            nb::arg("arrays")
        )
        .def(
            "fill_boundary",
            [](MultiFab& mf, const Geometry& geom) { mf.FillBoundary(geom.periodicity()); },
            nb::arg("geom")
        )
        .def(
            "fill_domain_boundary",
            [](MultiFab& mf,
               const Geometry& geom,
               nb::list bc_types_list,
               nb::list bc_values_list)
            {
                using namespace amrex;

                const Box& domain = geom.Domain();
                auto is_per = geom.isPeriodic();
                int nc = mf.nComp();
                int ng = mf.nGrow();

                // Parse bc_types: 6 ints (lo_x, hi_x, lo_y, hi_y, lo_z, hi_z)
                // 0=skip, 1=dirichlet, 2=neumann
                GpuArray<int, 6> bc_types;
                for (int i = 0; i < 6; ++i)
                    bc_types[i] = nb::cast<int>(bc_types_list[i]);

                // Parse bc_values: 6 lists of ncomp doubles
                // bc_values[face][comp] — wall value for dirichlet
                GpuArray<GpuArray<Real, AMREX_SPACEDIM>, 6> bc_vals{};
                for (int f = 0; f < 6; ++f)
                {
                    auto vals = nb::cast<nb::list>(bc_values_list[f]);
                    for (int c = 0; c < std::min(nc, (int)nb::len(vals)); ++c)
                        bc_vals[f][c] = nb::cast<Real>(vals[c]);
                }

                bool onDevice =
                    mf.arena()->isDeviceAccessible() && !mf.arena()->isHostAccessible();

                for (MFIter mfi(mf); mfi.isValid(); ++mfi)
                {
                    const Box& vbx = mfi.validbox();
                    auto arr = mf[mfi].array();

                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        // Low side
                        if (!is_per[d] && vbx.smallEnd(d) == domain.smallEnd(d)
                            && bc_types[2 * d] != 0)
                        {
                            int bct = bc_types[2 * d];
                            auto bv = bc_vals[2 * d];

                            for (int g = 0; g < ng; ++g)
                            {
                                int ghost_idx = vbx.smallEnd(d) - 1 - g;
                                int interior_idx = vbx.smallEnd(d) + g;

                                Box ghost_bx = vbx;
                                ghost_bx.setSmall(d, ghost_idx);
                                ghost_bx.setBig(d, ghost_idx);

                                if (onDevice)
                                {
                                    int dir = d;
                                    int gi = ghost_idx;
                                    int ii = interior_idx;
                                    ParallelFor(
                                        ghost_bx, nc,
                                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                                        {
                                            int ic = i, jc = j, kc = k;
                                            if (dir == 0) ic = ii;
                                            else if (dir == 1) jc = ii;
                                            else kc = ii;
                                            Real interior = arr(ic, jc, kc, n);
                                            if (bct == 1)
                                                arr(i, j, k, n) =
                                                    2.0 * bv[n] - interior;
                                            else
                                                arr(i, j, k, n) = interior;
                                        });
                                }
                                else
                                {
                                    const auto lo = lbound(ghost_bx);
                                    const auto hi = ubound(ghost_bx);
                                    for (int n = 0; n < nc; ++n)
                                        for (int k = lo.z; k <= hi.z; ++k)
                                            for (int j = lo.y; j <= hi.y; ++j)
                                                for (int i = lo.x; i <= hi.x; ++i)
                                                {
                                                    int ic = i, jc = j, kc = k;
                                                    if (d == 0) ic = interior_idx;
                                                    else if (d == 1) jc = interior_idx;
                                                    else kc = interior_idx;
                                                    Real interior = arr(ic, jc, kc, n);
                                                    if (bct == 1)
                                                        arr(i, j, k, n) =
                                                            2.0 * bv[n] - interior;
                                                    else
                                                        arr(i, j, k, n) = interior;
                                                }
                                }
                            }
                        }

                        // High side
                        if (!is_per[d] && vbx.bigEnd(d) == domain.bigEnd(d)
                            && bc_types[2 * d + 1] != 0)
                        {
                            int bct = bc_types[2 * d + 1];
                            auto bv = bc_vals[2 * d + 1];

                            for (int g = 0; g < ng; ++g)
                            {
                                int ghost_idx = vbx.bigEnd(d) + 1 + g;
                                int interior_idx = vbx.bigEnd(d) - g;

                                Box ghost_bx = vbx;
                                ghost_bx.setSmall(d, ghost_idx);
                                ghost_bx.setBig(d, ghost_idx);

                                if (onDevice)
                                {
                                    int dir = d;
                                    int gi = ghost_idx;
                                    int ii = interior_idx;
                                    ParallelFor(
                                        ghost_bx, nc,
                                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                                        {
                                            int ic = i, jc = j, kc = k;
                                            if (dir == 0) ic = ii;
                                            else if (dir == 1) jc = ii;
                                            else kc = ii;
                                            Real interior = arr(ic, jc, kc, n);
                                            if (bct == 1)
                                                arr(i, j, k, n) =
                                                    2.0 * bv[n] - interior;
                                            else
                                                arr(i, j, k, n) = interior;
                                        });
                                }
                                else
                                {
                                    const auto lo = lbound(ghost_bx);
                                    const auto hi = ubound(ghost_bx);
                                    for (int n = 0; n < nc; ++n)
                                        for (int k = lo.z; k <= hi.z; ++k)
                                            for (int j = lo.y; j <= hi.y; ++j)
                                                for (int i = lo.x; i <= hi.x; ++i)
                                                {
                                                    int ic = i, jc = j, kc = k;
                                                    if (d == 0) ic = interior_idx;
                                                    else if (d == 1) jc = interior_idx;
                                                    else kc = interior_idx;
                                                    Real interior = arr(ic, jc, kc, n);
                                                    if (bct == 1)
                                                        arr(i, j, k, n) =
                                                            2.0 * bv[n] - interior;
                                                    else
                                                        arr(i, j, k, n) = interior;
                                                }
                                }
                            }
                        }
                    }
                }

                if (onDevice)
                    Gpu::streamSynchronize();
            },
            nb::arg("geom"),
            nb::arg("bc_types"),
            nb::arg("bc_values")
        );
}
