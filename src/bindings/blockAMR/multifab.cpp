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

namespace nb = nanobind;

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
        .def("valid_box", [](MFIterator& self) { return self.mfi->validbox(); });

    nb::class_<MultiFab>(m, "MultiFab")
        .def(
            nb::init<const BoxArray&, const DistributionMapping&, int, int>(),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("ncomp"),
            nb::arg("ngrow")
        )
        .def("num_comp", &MultiFab::nComp)
        .def("n_grow", [](const MultiFab& mf) { return mf.nGrow(); })
        .def(
            "array",
            [](nb::object self, MFIterator& mfi)
            {
                MultiFab& mf = nb::cast<MultiFab&>(self);
                auto& fab = mf[mfi.get()];
                const Box& bx = mfi.get().validbox();
                auto lo = bx.smallEnd();
                auto hi = bx.bigEnd();
                int nx = hi[0] - lo[0] + 1;
                int ny = hi[1] - lo[1] + 1;
                int nz = hi[2] - lo[2] + 1;
                int nc = mf.nComp();

                // Compute offset from FAB start to valid region in Fortran order
                Real* ptr = fab.dataPtr();
                const Box& fabbox = fab.box();
                int ngx = lo[0] - fabbox.smallEnd(0);
                int ngy = lo[1] - fabbox.smallEnd(1);
                int ngz = lo[2] - fabbox.smallEnd(2);
                int fnx = fabbox.length(0);
                int fny = fabbox.length(1);
                int fnz = fabbox.length(2);

                Real* validPtr = ptr + ngx + ngy * fnx + ngz * fnx * fny;

                size_t shape[4] = {(size_t)nx, (size_t)ny, (size_t)nz, (size_t)nc};
                // Fortran-order strides in elements: x fastest, component outermost
                int64_t strides[4] = {
                    1, (int64_t)fnx, (int64_t)(fnx * fny), (int64_t)(fnx * fny * fnz)
                };

                return nb::ndarray<nb::numpy, Real, nb::ndim<4>>(validPtr, 4, shape, self, strides);
            },
            nb::arg("mfi")
        )
        .def(
            "grown_array",
            [](nb::object self, MFIterator& mfi)
            {
                MultiFab& mf = nb::cast<MultiFab&>(self);
                auto& fab = mf[mfi.get()];
                const Box& bx = fab.box(); // grown box
                int nx = bx.length(0);
                int ny = bx.length(1);
                int nz = bx.length(2);
                int nc = mf.nComp();
                Real* ptr = fab.dataPtr();

                size_t shape[4] = {(size_t)nx, (size_t)ny, (size_t)nz, (size_t)nc};
                // Fortran-order strides in elements: x fastest, component outermost
                int64_t strides[4] = {1, (int64_t)nx, (int64_t)(nx * ny), (int64_t)(nx * ny * nz)};

                return nb::ndarray<nb::numpy, Real, nb::ndim<4>>(ptr, 4, shape, self, strides);
            },
            nb::arg("mfi")
        )
        .def(
            "fill_boundary",
            [](MultiFab& mf, const Geometry& geom) { mf.FillBoundary(geom.periodicity()); },
            nb::arg("geom")
        );
}
