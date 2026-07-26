// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <AMReX_TagBox.H>
#include <AMReX_MultiFab.H>
#include <AMReX_GpuLaunch.H>

#include <memory>

namespace nb = nanobind;

// Python-friendly iterator for TagBoxArray (mirrors MFIterator for MultiFab)
struct TagBoxIterator
{
    amrex::TagBoxArray* tba;
    std::unique_ptr<amrex::MFIter> mfi;
    bool needsAdvance;

    explicit TagBoxIterator(amrex::TagBoxArray& tba_)
        : tba(&tba_), mfi(nullptr), needsAdvance(false)
    {}
};

void registerTagBox(nb::module_& m)
{
    using namespace amrex;

    m.attr("TAG_CLEAR") = static_cast<int>(TagBox::CLEAR);
    m.attr("TAG_SET") = static_cast<int>(TagBox::SET);

    nb::class_<TagBox>(m, "TagBox").def("box", [](const TagBox& tb) { return tb.box(); });

    nb::class_<TagBoxIterator>(m, "TagBoxIterator")
        .def(
            "__init__",
            [](TagBoxIterator* self, TagBoxArray& tba) { new (self) TagBoxIterator(tba); },
            nb::arg("tba"),
            nb::keep_alive<1, 2>()
        )
        .def(
            "__iter__",
            [](TagBoxIterator& self) -> TagBoxIterator&
            {
                self.mfi = std::make_unique<MFIter>(*self.tba);
                self.needsAdvance = false;
                return self;
            },
            nb::rv_policy::reference
        )
        .def(
            "__next__",
            [](nb::object pySelf) -> nb::object
            {
                TagBoxIterator& self = nb::cast<TagBoxIterator&>(pySelf);
                if (self.needsAdvance) ++(*self.mfi);
                if (!self.mfi || !self.mfi->isValid())
                {
                    self.mfi.reset();
                    throw nb::stop_iteration();
                }
                self.needsAdvance = true;
                return pySelf;
            }
        )
        .def("valid_box", [](TagBoxIterator& self) { return self.mfi->validbox(); })
        .def(
            "tag_box",
            [](TagBoxIterator& self) -> TagBox& { return (*self.tba)[*self.mfi]; },
            nb::rv_policy::reference_internal
        )
        .def(
            "set_tags",
            [](TagBoxIterator& self, nb::ndarray<nb::ro> mask)
            {
                // mask is an int array (nx, ny, nz) — nonzero means TAG_SET
                // Accepts both host (numpy) and device (JAX) arrays.
                auto& tb = (*self.tba)[*self.mfi];
                const Box& vbx = self.mfi->validbox();
                auto tag4 = tb.array();
                const auto lo = amrex::lbound(vbx);
                const auto hi = amrex::ubound(vbx);
                int nx = hi.x - lo.x + 1;
                int ny = hi.y - lo.y + 1;
                const Long npts = vbx.numPts();

                bool srcOnDevice = (mask.device_type() != nb::device::cpu::value);

                if (srcOnDevice)
                {
                    // Source is on GPU — read directly, no host copy
                    const int* mdata = static_cast<const int*>(mask.data());
                    amrex::ParallelFor(
                        vbx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            int li = i - lo.x;
                            int lj = j - lo.y;
                            int lk = k - lo.z;
                            Long idx = li + (Long)lj * nx + (Long)lk * nx * ny;
                            if (mdata[idx] != 0)
                                tag4(i, j, k) = TagBox::SET;
                        }
                    );
                    Gpu::streamSynchronize();
                }
                else
                {
                    // Source is on host — convert int→char, copy to device
                    const int* mdata = static_cast<const int*>(mask.data());
                    char* tagBuf =
                        static_cast<char*>(The_Arena()->alloc(npts * sizeof(char)));
                    char* hostBuf = new char[npts];
                    for (Long n = 0; n < npts; ++n)
                        hostBuf[n] = (mdata[n] != 0) ? TagBox::SET : TagBox::CLEAR;

                    Gpu::htod_memcpy_async(tagBuf, hostBuf, npts * sizeof(char));
                    Gpu::streamSynchronize();

                    const char* src = tagBuf;
                    amrex::ParallelFor(
                        vbx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            int li = i - lo.x;
                            int lj = j - lo.y;
                            int lk = k - lo.z;
                            Long idx = li + (Long)lj * nx + (Long)lk * nx * ny;
                            if (src[idx] == TagBox::SET)
                                tag4(i, j, k) = TagBox::SET;
                        }
                    );
                    Gpu::streamSynchronize();

                    The_Arena()->free(tagBuf);
                    delete[] hostBuf;
                }
            },
            nb::arg("mask")
        );

    nb::class_<TagBoxArray>(m, "TagBoxArray")
        .def(
            "__getitem__",
            [](TagBoxArray& tba, const MFIter& mfi) -> TagBox& { return tba[mfi]; },
            nb::arg("mfi"),
            nb::rv_policy::reference_internal
        )
        .def(
            "set_tags",
            [](TagBoxArray& tba, nb::object mfi_obj, nb::ndarray<nb::ro> mask)
            {
                // Extract amrex::MFIter from the Python MFIterator via .get()
                auto& mfi_inner = nb::cast<MFIter&>(mfi_obj.attr("get")());
                auto& tb = tba[mfi_inner];
                const Box& vbx = mfi_inner.validbox();
                auto tag4 = tb.array();
                const auto lo = amrex::lbound(vbx);
                const auto hi = amrex::ubound(vbx);
                int nx = hi.x - lo.x + 1;
                int ny = hi.y - lo.y + 1;
                int nz = hi.z - lo.z + 1;
                const Long npts = vbx.numPts();

                // Detect memory layout: Fortran (col-major) vs C (row-major)
                bool isFortran = false;
                if (mask.ndim() >= 2)
                    isFortran = (mask.stride(0) <= mask.stride(mask.ndim() - 1));

                bool srcOnDevice = (mask.device_type() != nb::device::cpu::value);

                if (srcOnDevice)
                {
                    const int* mdata = static_cast<const int*>(mask.data());
                    amrex::ParallelFor(
                        vbx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            int li = i - lo.x;
                            int lj = j - lo.y;
                            int lk = k - lo.z;
                            Long idx = isFortran
                                ? li + (Long)lj * nx + (Long)lk * nx * ny
                                : (Long)li * ny * nz + (Long)lj * nz + lk;
                            if (mdata[idx] != 0)
                                tag4(i, j, k) = TagBox::SET;
                        }
                    );
                    Gpu::streamSynchronize();
                }
                else
                {
                    const int* mdata = static_cast<const int*>(mask.data());
                    char* tagBuf =
                        static_cast<char*>(The_Arena()->alloc(npts * sizeof(char)));
                    char* hostBuf = new char[npts];
                    for (int li = 0; li < nx; ++li)
                        for (int lj = 0; lj < ny; ++lj)
                            for (int lk = 0; lk < nz; ++lk)
                            {
                                Long src_idx = isFortran
                                    ? li + (Long)lj * nx + (Long)lk * nx * ny
                                    : (Long)li * ny * nz + (Long)lj * nz + lk;
                                Long dst_idx = li + (Long)lj * nx + (Long)lk * nx * ny;
                                hostBuf[dst_idx] = (mdata[src_idx] != 0)
                                    ? TagBox::SET : TagBox::CLEAR;
                            }
                    Gpu::htod_memcpy_async(tagBuf, hostBuf, npts * sizeof(char));
                    Gpu::streamSynchronize();
                    const char* src = tagBuf;
                    amrex::ParallelFor(
                        vbx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            int li = i - lo.x;
                            int lj = j - lo.y;
                            int lk = k - lo.z;
                            Long idx = li + (Long)lj * nx + (Long)lk * nx * ny;
                            if (src[idx] == TagBox::SET)
                                tag4(i, j, k) = TagBox::SET;
                        }
                    );
                    Gpu::streamSynchronize();
                    The_Arena()->free(tagBuf);
                    delete[] hostBuf;
                }
            },
            nb::arg("mfi"),
            nb::arg("mask")
        );
}
