// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <AMReX_BoxArray.H>
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
