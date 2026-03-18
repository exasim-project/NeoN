// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include <AMReX_Box.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealBox.H>

namespace nb = nanobind;

void registerBox(nb::module_& m)
{
    using namespace amrex;

    nb::class_<IntVect>(m, "IntVect")
        .def(nb::init<int, int, int>(), nb::arg("x"), nb::arg("y"), nb::arg("z"))
        .def("__getitem__", [](const IntVect& iv, int i) { return iv[i]; });

    nb::class_<Box>(m, "Box")
        .def("__init__",
             [](Box* self, const std::array<int, 3>& lo, const std::array<int, 3>& hi) {
                 new (self) Box(IntVect(lo[0], lo[1], lo[2]), IntVect(hi[0], hi[1], hi[2]));
             },
             nb::arg("small"), nb::arg("big"))
        .def(nb::init<const IntVect&, const IntVect&>(), nb::arg("small"), nb::arg("big"))
        .def(nb::init<const IntVect&, const IntVect&, IndexType>(),
             nb::arg("small"), nb::arg("big"), nb::arg("t"))
        .def(
            "__init__",
            [](Box* self, const std::array<int, 3>& lo, const std::array<int, 3>& hi,
               IndexType t) {
                new (self) Box(IntVect(lo[0], lo[1], lo[2]), IntVect(hi[0], hi[1], hi[2]), t);
            },
            nb::arg("small"), nb::arg("big"), nb::arg("t"))
        .def("ix_type", &Box::ixType)
        .def("cell_centered", &Box::cellCentered)
        .def(
            "convert",
            [](Box& bx, IndexType t) -> Box& { return bx.convert(t); },
            nb::arg("typ"), nb::rv_policy::reference)
        .def(
            "convert",
            [](Box& bx, const IntVect& t) -> Box& { return bx.convert(t); },
            nb::arg("typ"), nb::rv_policy::reference)
        .def(
            "surrounding_nodes",
            [](Box& bx) -> Box& { return bx.surroundingNodes(); },
            nb::rv_policy::reference)
        .def(
            "surrounding_nodes",
            [](Box& bx, int dir) -> Box& { return bx.surroundingNodes(dir); },
            nb::arg("dir"), nb::rv_policy::reference)
        .def(
            "enclosed_cells",
            [](Box& bx) -> Box& { return bx.enclosedCells(); },
            nb::rv_policy::reference)
        .def(
            "enclosed_cells",
            [](Box& bx, int dir) -> Box& { return bx.enclosedCells(dir); },
            nb::arg("dir"), nb::rv_policy::reference)
        .def("small_end",
             [](const Box& bx) {
                 auto se = bx.smallEnd();
                 return std::array<int, 3>{se[0], se[1], se[2]};
             })
        .def("big_end",
             [](const Box& bx) {
                 auto be = bx.bigEnd();
                 return std::array<int, 3>{be[0], be[1], be[2]};
             })
        .def("num_pts", &Box::numPts);

    nb::class_<RealBox>(m, "RealBox")
        .def("__init__",
             [](RealBox* self, const std::array<double, 3>& lo,
                const std::array<double, 3>& hi) {
                 new (self) RealBox(lo, hi);
             },
             nb::arg("lo"), nb::arg("hi"));
}
