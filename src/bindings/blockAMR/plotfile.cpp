// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Vector.H>

namespace nb = nanobind;

void registerPlotfile(nb::module_& m)
{
    using namespace amrex;

    m.def(
        "write_single_level_plotfile",
        [](const std::string& name, const MultiFab& mf,
           const std::vector<std::string>& varnames, const Geometry& geom, double time,
           int step) {
            Vector<std::string> vn(varnames.begin(), varnames.end());
            WriteSingleLevelPlotfile(name, mf, vn, geom, time, step);
        },
        nb::arg("plotfilename"), nb::arg("mf"), nb::arg("varnames"), nb::arg("geom"),
        nb::arg("time"), nb::arg("level_step"));
}
