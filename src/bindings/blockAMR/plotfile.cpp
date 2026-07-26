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

    // Helper: combine a list of MultiFabs into one (concatenating components)
    auto combineMFs = [](nb::list mf_list) -> MultiFab
    {
        int total_ncomp = 0;
        for (size_t i = 0; i < nb::len(mf_list); ++i)
            total_ncomp += nb::cast<const MultiFab&>(mf_list[i]).nComp();

        const auto& first = nb::cast<const MultiFab&>(mf_list[0]);
        MultiFab combined(first.boxArray(), first.DistributionMap(),
                          total_ncomp, 0);
        int dst = 0;
        for (size_t i = 0; i < nb::len(mf_list); ++i)
        {
            const auto& src = nb::cast<const MultiFab&>(mf_list[i]);
            MultiFab::Copy(combined, src, 0, dst, src.nComp(), 0);
            dst += src.nComp();
        }
        return combined;
    };

    // Single MultiFab
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

    // List of MultiFabs (combined into one)
    m.def(
        "write_single_level_plotfile",
        [combineMFs](const std::string& name,
           nb::list mf_list,
           const std::vector<std::string>& varnames,
           const Geometry& geom,
           double time,
           int step)
        {
            MultiFab combined = combineMFs(mf_list);
            Vector<std::string> vn(varnames.begin(), varnames.end());
            WriteSingleLevelPlotfile(name, combined, vn, geom, time, step);
        },
        nb::arg("plotfilename"),
        nb::arg("mf_list"),
        nb::arg("varnames"),
        nb::arg("geom"),
        nb::arg("time"),
        nb::arg("level_step")
    );

    // Single MultiFab per level
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

    // List of MultiFabs per level (combined into one per level)
    m.def(
        "write_multilevel_plotfile",
        [combineMFs](const std::string& name,
           int nlevels,
           nb::list mf_lists,
           const std::vector<std::string>& varnames,
           nb::list geom_list,
           double time,
           const std::vector<int>& steps,
           nb::list rr_list)
        {
            // mf_lists is a list of lists: mf_lists[lev] = [mf_a, mf_b, ...]
            Vector<MultiFab> combined_storage;
            combined_storage.reserve(nlevels);
            Vector<const MultiFab*> mf_vec;
            for (int lev = 0; lev < nlevels; ++lev)
            {
                nb::list lev_list = nb::cast<nb::list>(mf_lists[lev]);
                combined_storage.push_back(combineMFs(lev_list));
                mf_vec.push_back(&combined_storage.back());
            }

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
        nb::arg("mf_lists"),
        nb::arg("varnames"),
        nb::arg("geom"),
        nb::arg("time"),
        nb::arg("level_steps"),
        nb::arg("ref_ratio")
    );
}
