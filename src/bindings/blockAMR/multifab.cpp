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

#include <cstring>

namespace nb = nanobind;

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
        .def("num_comp", &MultiFab::nComp)
        .def("n_grow", [](const MultiFab& mf) { return mf.nGrow(); })
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
            "copy_from",
            [](MultiFab& mf, MFIterator& mfi, nb::ndarray<nb::ro> src)
            {
                auto& fab = mf[mfi.get()];
                const Box& bx = mfi.get().validbox();
                int ncomp = (src.ndim() == 4) ? static_cast<int>(src.shape(3)) : 1;
                int nx = bx.length(0);
                int ny = bx.length(1);
                int nz = bx.length(2);
                size_t nbytes = (size_t)bx.numPts() * ncomp * sizeof(Real);

                const Real* srcPtr = static_cast<const Real*>(src.data());
                bool srcOnDevice = (src.device_type() != nb::device::cpu::value);
                bool dstOnDevice =
                    mf.arena()->isDeviceAccessible() && !mf.arena()->isHostAccessible();

                // Check if source is Fortran-order (column-major).
                // Fortran-order means x varies fastest, matching AMReX internal layout.
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
                    amrex::Gpu::htod_memcpy(devSrc, srcPtr, nbytes);
                }
                else
                {
                    devSrc = static_cast<Real*>(The_Pinned_Arena()->alloc(nbytes));
                    ownDevSrc = true;
                    amrex::Gpu::dtoh_memcpy(devSrc, srcPtr, nbytes);
                }

                if (srcIsFortran)
                {
                    // Source is Fortran-order — same layout as AMReX FAB, direct copy
                    auto* dstPtr = fab.dataPtr();
                    // Compute valid-region offset in the FAB
                    const auto fabBox = fab.box();
                    const auto fabLo = fabBox.smallEnd();
                    const auto bxLo = bx.smallEnd();
                    if (fabBox == bx)
                    {
                        // Valid box matches FAB box — direct memcpy
                        if (dstOnDevice)
                        {
                            amrex::Gpu::dtod_memcpy_async(dstPtr, devSrc, nbytes);
                            amrex::Gpu::streamSynchronize();
                        }
                        else
                            std::memcpy(dstPtr, devSrc, nbytes);
                    }
                    else
                    {
                        // FAB has ghost cells — copy per-component slab via Array4
                        auto arr4 = fab.array();
                        const auto lo = bx.smallEnd();
                        const Real* fSrc = devSrc;
                        if (dstOnDevice)
                        {
                            amrex::ParallelFor(
                                bx,
                                ncomp,
                                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                                {
                                    int li = i - lo[0];
                                    int lj = j - lo[1];
                                    int lk = k - lo[2];
                                    // F-order: leftmost index varies fastest
                                    size_t fIdx = (size_t)li + (size_t)lj * nx
                                                + (size_t)lk * nx * ny
                                                + (size_t)n * nx * ny * nz;
                                    arr4(i, j, k, n) = fSrc[fIdx];
                                }
                            );
                            amrex::Gpu::streamSynchronize();
                        }
                        else
                        {
                            const auto lo3 = amrex::lbound(bx);
                            const auto hi3 = amrex::ubound(bx);
                            for (int n = 0; n < ncomp; ++n)
                                for (int k = lo3.z; k <= hi3.z; ++k)
                                    for (int j = lo3.y; j <= hi3.y; ++j)
                                        for (int i = lo3.x; i <= hi3.x; ++i)
                                        {
                                            int li = i - lo[0];
                                            int lj = j - lo[1];
                                            int lk = k - lo[2];
                                            size_t fIdx = (size_t)li + (size_t)lj * nx
                                                        + (size_t)lk * nx * ny
                                                        + (size_t)n * nx * ny * nz;
                                            arr4(i, j, k, n) = fSrc[fIdx];
                                        }
                        }
                    }
                }
                else
                {
                    // Source is C-order — transpose to Fortran-order (AMReX layout)
                    auto arr4 = fab.array();
                    const auto lo = bx.smallEnd();
                    const Real* cSrc = devSrc;
                    if (dstOnDevice)
                    {
                        amrex::ParallelFor(
                            bx,
                            ncomp,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                int li = i - lo[0];
                                int lj = j - lo[1];
                                int lk = k - lo[2];
                                // C-order: rightmost index varies fastest
                                size_t cIdx = ((size_t)n * nx * ny * nz) + ((size_t)li * ny * nz)
                                            + ((size_t)lj * nz) + lk;
                                arr4(i, j, k, n) = cSrc[cIdx];
                            }
                        );
                        amrex::Gpu::streamSynchronize();
                    }
                    else
                    {
                        const auto lo3 = amrex::lbound(bx);
                        const auto hi3 = amrex::ubound(bx);
                        for (int n = 0; n < ncomp; ++n)
                            for (int k = lo3.z; k <= hi3.z; ++k)
                                for (int j = lo3.y; j <= hi3.y; ++j)
                                    for (int i = lo3.x; i <= hi3.x; ++i)
                                    {
                                        int li = i - lo[0];
                                        int lj = j - lo[1];
                                        int lk = k - lo[2];
                                        size_t cIdx = ((size_t)n * nx * ny * nz)
                                                    + ((size_t)li * ny * nz) + ((size_t)lj * nz) + lk;
                                        arr4(i, j, k, n) = cSrc[cIdx];
                                    }
                    }
                }

                if (ownDevSrc)
                {
                    if (dstOnDevice) mf.arena()->free(devSrc);
                    else
                        The_Pinned_Arena()->free(devSrc);
                }
            },
            nb::arg("mfi"),
            nb::arg("src")
        )
        .def(
            "fill_boundary",
            [](MultiFab& mf, const Geometry& geom) { mf.FillBoundary(geom.periodicity()); },
            nb::arg("geom")
        );
}
