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
        [](const std::string& name,
           const MultiFab& mf,
           const std::vector<std::string>& varnames,
           const Geometry& geom,
           double time,
           int step)
        {
            Vector<std::string> vn(varnames.begin(), varnames.end());
            WriteSingleLevelPlotfile(name, mf, vn, geom, time, step);
        },
        nb::arg("plotfilename"),
        nb::arg("mf"),
        nb::arg("varnames"),
        nb::arg("geom"),
        nb::arg("time"),
        nb::arg("level_step")
    );

    m.def(
        "write_multilevel_plotfile",
        [](const std::string& name,
           int nlevels,
           nb::list mf_list,
           const std::vector<std::string>& varnames,
           nb::list geom_list,
           double time,
           const std::vector<int>& steps,
           nb::list rr_list)
        {
            Vector<const MultiFab*> mf_vec;
            for (size_t i = 0; i < nb::len(mf_list); ++i)
                mf_vec.push_back(&nb::cast<const MultiFab&>(mf_list[i]));

            Vector<std::string> vn(varnames.begin(), varnames.end());

            Vector<Geometry> geom_vec;
            for (size_t i = 0; i < nb::len(geom_list); ++i)
                geom_vec.push_back(nb::cast<Geometry>(geom_list[i]));

            Vector<int> steps_vec(steps.begin(), steps.end());

            Vector<IntVect> rr_vec;
            for (size_t i = 0; i < nb::len(rr_list); ++i)
                rr_vec.push_back(nb::cast<IntVect>(rr_list[i]));

            WriteMultiLevelPlotfile(name, nlevels, mf_vec, vn, geom_vec, time, steps_vec, rr_vec);
        },
        nb::arg("plotfilename"),
        nb::arg("nlevels"),
        nb::arg("mf"),
        nb::arg("varnames"),
        nb::arg("geom"),
        nb::arg("time"),
        nb::arg("level_steps"),
        nb::arg("ref_ratio")
    );
}
