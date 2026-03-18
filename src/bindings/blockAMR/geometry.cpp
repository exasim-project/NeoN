// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include <AMReX_Geometry.H>
#include <AMReX_CoordSys.H>
#include <AMReX_Periodicity.H>
#include <AMReX_RealBox.H>

namespace nb = nanobind;

void registerGeometry(nb::module_& m)
{
    using namespace amrex;

    nb::class_<Periodicity>(m, "Periodicity");

    nb::class_<Geometry>(m, "Geometry")
        .def("__init__",
             [](Geometry* self, const Box& dom, const RealBox& rb, int coord,
                const std::array<int, 3>& isPer) {
                 Array<int, AMREX_SPACEDIM> per = {isPer[0], isPer[1], isPer[2]};
                 new (self) Geometry(dom, rb, coord, per);
             },
             nb::arg("dom"), nb::arg("rb"), nb::arg("coord"), nb::arg("is_per"))
        .def("cell_size",
             [](const Geometry& geom) {
                 return std::array<double, 3>{geom.CellSize(0), geom.CellSize(1), geom.CellSize(2)};
             })
        .def("periodicity",
             [](const Geometry& g) { return g.periodicity(); })
        .def("prob_lo",
             [](const Geometry& g) {
                 return std::array<double, 3>{g.ProbLo(0), g.ProbLo(1), g.ProbLo(2)};
             })
        .def("prob_hi", [](const Geometry& g) {
            return std::array<double, 3>{g.ProbHi(0), g.ProbHi(1), g.ProbHi(2)};
        });
}
