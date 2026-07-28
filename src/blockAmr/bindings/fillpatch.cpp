// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <AMReX_BCRec.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_PhysBCFunct.H>

namespace nb = nanobind;

void registerFillPatch(nb::module_& m)
{
    using namespace amrex;

    // BCRec
    nb::class_<BCRec>(m, "BCRec")
        .def(
            "__init__",
            [](BCRec* self, int loX, int loY, int loZ, int hiX, int hiY, int hiZ)
            { new (self) BCRec(AMREX_D_DECL(loX, loY, loZ), AMREX_D_DECL(hiX, hiY, hiZ)); },
            nb::arg("lo_x"),
            nb::arg("lo_y"),
            nb::arg("lo_z"),
            nb::arg("hi_x"),
            nb::arg("hi_y"),
            nb::arg("hi_z")
        );

    m.def(
        "periodic_bcrec",
        []()
        {
            return BCRec(
                AMREX_D_DECL(BCType::int_dir, BCType::int_dir, BCType::int_dir),
                AMREX_D_DECL(BCType::int_dir, BCType::int_dir, BCType::int_dir)
            );
        }
    );

    // Interpolater singletons
    nb::class_<Interpolater>(m, "Interpolater");

    m.def(
        "cell_cons_interp",
        []() -> Interpolater* { return &cell_cons_interp; },
        nb::rv_policy::reference
    );

    m.def(
        "pc_interp", []() -> Interpolater* { return &pc_interp; }, nb::rv_policy::reference
    );

    // FillPatchSingleLevel (periodic only, PhysBCFunctNoOp)
    m.def(
        "fill_patch_single_level",
        [](MultiFab& mf,
           Real time,
           nb::list smf_list,
           nb::list stime_list,
           const Geometry& geom,
           int scomp,
           int ncomp)
        {
            Vector<MultiFab*> smf_vec;
            for (size_t i = 0; i < nb::len(smf_list); ++i)
                smf_vec.push_back(&nb::cast<MultiFab&>(smf_list[i]));

            Vector<Real> stime_vec;
            for (size_t i = 0; i < nb::len(stime_list); ++i)
                stime_vec.push_back(nb::cast<Real>(stime_list[i]));

            PhysBCFunctNoOp bc;
            FillPatchSingleLevel(mf, time, smf_vec, stime_vec, scomp, 0, ncomp, geom, bc, 0);
        },
        nb::arg("mf"),
        nb::arg("time"),
        nb::arg("smf"),
        nb::arg("stime"),
        nb::arg("geom"),
        nb::arg("scomp"),
        nb::arg("ncomp")
    );

    // InterpFromCoarseLevel (coarse-only interpolation, no fine source needed)
    m.def(
        "interp_from_coarse_level",
        [](MultiFab& mf,
           Real time,
           const MultiFab& cmf,
           int scomp,
           int dcomp,
           int ncomp,
           const Geometry& cgeom,
           const Geometry& fgeom,
           const IntVect& ratio,
           Interpolater* mapper,
           nb::list bcs_list)
        {
            Vector<BCRec> bcs_vec;
            for (size_t i = 0; i < nb::len(bcs_list); ++i)
                bcs_vec.push_back(nb::cast<BCRec>(bcs_list[i]));

            PhysBCFunctNoOp cbc, fbc;
            InterpFromCoarseLevel(
                mf,
                time,
                cmf,
                scomp,
                dcomp,
                ncomp,
                cgeom,
                fgeom,
                cbc,
                0,
                fbc,
                0,
                ratio,
                mapper,
                bcs_vec,
                0
            );
        },
        nb::arg("mf"),
        nb::arg("time"),
        nb::arg("cmf"),
        nb::arg("scomp"),
        nb::arg("dcomp"),
        nb::arg("ncomp"),
        nb::arg("cgeom"),
        nb::arg("fgeom"),
        nb::arg("ratio"),
        nb::arg("mapper"),
        nb::arg("bcs")
    );

    // FillPatchTwoLevels (periodic only, PhysBCFunctNoOp)
    m.def(
        "fill_patch_two_levels",
        [](MultiFab& mf,
           Real time,
           nb::list cmf_list,
           nb::list ct_list,
           nb::list fmf_list,
           nb::list ft_list,
           int scomp,
           int dcomp,
           int ncomp,
           const Geometry& cgeom,
           const Geometry& fgeom,
           const IntVect& ratio,
           Interpolater* mapper,
           nb::list bcs_list)
        {
            Vector<MultiFab*> cmf_vec;
            for (size_t i = 0; i < nb::len(cmf_list); ++i)
                cmf_vec.push_back(&nb::cast<MultiFab&>(cmf_list[i]));

            Vector<Real> ct_vec;
            for (size_t i = 0; i < nb::len(ct_list); ++i)
                ct_vec.push_back(nb::cast<Real>(ct_list[i]));

            Vector<MultiFab*> fmf_vec;
            for (size_t i = 0; i < nb::len(fmf_list); ++i)
                fmf_vec.push_back(&nb::cast<MultiFab&>(fmf_list[i]));

            Vector<Real> ft_vec;
            for (size_t i = 0; i < nb::len(ft_list); ++i)
                ft_vec.push_back(nb::cast<Real>(ft_list[i]));

            Vector<BCRec> bcs_vec;
            for (size_t i = 0; i < nb::len(bcs_list); ++i)
                bcs_vec.push_back(nb::cast<BCRec>(bcs_list[i]));

            PhysBCFunctNoOp cbc, fbc;
            FillPatchTwoLevels(
                mf,
                time,
                cmf_vec,
                ct_vec,
                fmf_vec,
                ft_vec,
                scomp,
                dcomp,
                ncomp,
                cgeom,
                fgeom,
                cbc,
                0,
                fbc,
                0,
                ratio,
                mapper,
                bcs_vec,
                0
            );
        },
        nb::arg("mf"),
        nb::arg("time"),
        nb::arg("cmf"),
        nb::arg("ct"),
        nb::arg("fmf"),
        nb::arg("ft"),
        nb::arg("scomp"),
        nb::arg("dcomp"),
        nb::arg("ncomp"),
        nb::arg("cgeom"),
        nb::arg("fgeom"),
        nb::arg("ratio"),
        nb::arg("mapper"),
        nb::arg("bcs")
    );
}
