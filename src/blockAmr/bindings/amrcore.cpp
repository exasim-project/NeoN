// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <AMReX_AmrCore.H>
#include <AMReX_AmrMesh.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_TagBox.H>

namespace nb = nanobind;

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)
template<>
struct is_copy_constructible<amrex::AmrCore> : std::false_type
{
};
template<>
struct is_copy_constructible<amrex::TagBoxArray> : std::false_type
{
};
NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

struct PyAmrCore : amrex::AmrCore
{
    NB_TRAMPOLINE(amrex::AmrCore, 5);

    void MakeNewLevelFromScratch(
        int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm
    ) override
    {
        NB_OVERRIDE_PURE_NAME(
            "make_new_level_from_scratch", MakeNewLevelFromScratch, lev, time, ba, dm
        );
    }

    void MakeNewLevelFromCoarse(
        int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm
    ) override
    {
        NB_OVERRIDE_PURE_NAME(
            "make_new_level_from_coarse", MakeNewLevelFromCoarse, lev, time, ba, dm
        );
    }

    void RemakeLevel(
        int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm
    ) override
    {
        NB_OVERRIDE_PURE_NAME("remake_level", RemakeLevel, lev, time, ba, dm);
    }

    void ClearLevel(int lev) override { NB_OVERRIDE_PURE_NAME("clear_level", ClearLevel, lev); }

    void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "error_est", true);
        nb::object pyTags = nb::cast(tags, nb::rv_policy::reference);
        nb_trampoline.base().attr(nb_ticket.key)(lev, pyTags, time, ngrow);
        // tags stays AMReX-owned: clear the Python instance's destruct flag or it double-frees.
        nb::detail::nb_inst_set_state(pyTags.ptr(), false, false);
    }
};

void registerAmrCore(nb::module_& m)
{
    using namespace amrex;

    nb::class_<AmrInfo>(m, "AmrInfo")
        .def(nb::init<>())
        .def_rw("max_level", &AmrInfo::max_level)
        .def_rw("grid_eff", &AmrInfo::grid_eff)
        .def(
            "set_ref_ratio",
            [](AmrInfo& ai, int lev, int r)
            {
                if (lev >= static_cast<int>(ai.ref_ratio.size()))
                    ai.ref_ratio.resize(lev + 1, IntVect(2));
                ai.ref_ratio[lev] = IntVect(AMREX_D_DECL(r, r, r));
            },
            nb::arg("lev"),
            nb::arg("ratio")
        )
        .def(
            "set_ref_ratio",
            [](AmrInfo& ai, int lev, const IntVect& r)
            {
                if (lev >= static_cast<int>(ai.ref_ratio.size()))
                    ai.ref_ratio.resize(lev + 1, IntVect(2));
                ai.ref_ratio[lev] = r;
            },
            nb::arg("lev"),
            nb::arg("ratio")
        )
        .def(
            "set_max_grid_size",
            [](AmrInfo& ai, int lev, int s)
            {
                if (lev >= static_cast<int>(ai.max_grid_size.size()))
                    ai.max_grid_size.resize(lev + 1, IntVect(32));
                ai.max_grid_size[lev] = IntVect(AMREX_D_DECL(s, s, s));
            },
            nb::arg("lev"),
            nb::arg("size")
        )
        .def(
            "set_blocking_factor",
            [](AmrInfo& ai, int lev, int b)
            {
                if (lev >= static_cast<int>(ai.blocking_factor.size()))
                    ai.blocking_factor.resize(lev + 1, IntVect(8));
                ai.blocking_factor[lev] = IntVect(AMREX_D_DECL(b, b, b));
            },
            nb::arg("lev"),
            nb::arg("factor")
        )
        .def(
            "set_n_error_buf",
            [](AmrInfo& ai, int lev, int n)
            {
                if (lev >= static_cast<int>(ai.n_error_buf.size()))
                    ai.n_error_buf.resize(lev + 1, IntVect(1));
                ai.n_error_buf[lev] = IntVect(AMREX_D_DECL(n, n, n));
            },
            nb::arg("lev"),
            nb::arg("buf")
        );

    nb::class_<AmrCore, PyAmrCore>(m, "AmrCore")
        .def(nb::init<const Geometry&, const AmrInfo&>(), nb::arg("geom"), nb::arg("amr_info"))
        .def(
            "init_from_scratch",
            [](AmrCore& ac, Real time) { ac.InitFromScratch(time); },
            nb::arg("time")
        )
        .def(
            "regrid",
            [](AmrCore& ac, int lbase, Real time) { ac.regrid(lbase, time); },
            nb::arg("lbase"),
            nb::arg("time")
        )
        .def_prop_ro("finest_level", [](const AmrCore& ac) { return ac.finestLevel(); })
        .def_prop_ro("max_level", [](const AmrCore& ac) { return ac.maxLevel(); })
        .def(
            "geom",
            [](const AmrCore& ac, int lev) -> const Geometry& { return ac.Geom(lev); },
            nb::arg("lev"),
            nb::rv_policy::reference_internal
        )
        .def(
            "box_array",
            [](const AmrCore& ac, int lev) -> const BoxArray& { return ac.boxArray(lev); },
            nb::arg("lev"),
            nb::rv_policy::reference_internal
        )
        .def(
            "dm",
            [](const AmrCore& ac, int lev) -> const DistributionMapping&
            { return ac.DistributionMap(lev); },
            nb::arg("lev"),
            nb::rv_policy::reference_internal
        )
        .def(
            "ref_ratio", [](const AmrCore& ac, int lev) { return ac.refRatio(lev); }, nb::arg("lev")
        );

    m.def(
        "average_down",
        [](const MultiFab& fine,
           MultiFab& crse,
           const Geometry& fgeom,
           const Geometry& cgeom,
           int scomp,
           int ncomp,
           const IntVect& ratio)
        { amrex::average_down(fine, crse, fgeom, cgeom, scomp, ncomp, ratio); },
        nb::arg("fine"),
        nb::arg("crse"),
        nb::arg("fine_geom"),
        nb::arg("crse_geom"),
        nb::arg("scomp"),
        nb::arg("ncomp"),
        nb::arg("ratio")
    );
}
