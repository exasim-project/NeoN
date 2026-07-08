// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>

#include <AMReX_Config.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBFabFactory.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_MultiCutFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>

#include "arenas.hpp"
#endif

namespace nb = nanobind;

void registerEBFactory(nb::module_& m)
{
#ifdef AMREX_USE_EB
    using namespace amrex;

    nb::class_<EBFArrayBoxFactory>(m, "EBFArrayBoxFactory")
        .def(
            "vol_frac",
            [](const EBFArrayBoxFactory& self) -> const MultiFab&
            { return self.getVolFrac(); },
            nb::rv_policy::reference_internal
        )
        .def(
            "level_set",
            [](const EBFArrayBoxFactory& self) -> const MultiFab&
            { return self.getLevelSet(); },
            nb::rv_policy::reference_internal
        )
        .def(
            "area_frac",
            [](const EBFArrayBoxFactory& self, int dir, Real regular, Real covered)
            {
                auto a = self.getAreaFrac();
                AMREX_ASSERT(dir >= 0 && dir < AMREX_SPACEDIM);
                return a[dir]->ToMultiFab(regular, covered);
            },
            nb::arg("dir"),
            nb::arg("regular") = Real(1.0),
            nb::arg("covered") = Real(0.0)
        )
        .def(
            "face_cent",
            [](const EBFArrayBoxFactory& self, int dir, Real regular, Real covered)
            {
                auto a = self.getFaceCent();
                AMREX_ASSERT(dir >= 0 && dir < AMREX_SPACEDIM);
                return a[dir]->ToMultiFab(regular, covered);
            },
            nb::arg("dir"),
            nb::arg("regular") = Real(0.0),
            nb::arg("covered") = Real(0.0)
        )
        .def("is_all_regular", &EBFArrayBoxFactory::isAllRegular)
        .def("max_coarsening_level", &EBFArrayBoxFactory::maxCoarseningLevel)
        .def(
            "box_array",
            [](const EBFArrayBoxFactory& self) -> const BoxArray&
            { return self.boxArray(); },
            nb::rv_policy::reference_internal
        )
        .def(
            "distribution_map",
            [](const EBFArrayBoxFactory& self) -> const DistributionMapping&
            { return self.DistributionMap(); },
            nb::rv_policy::reference_internal
        )
        .def(
            "geom",
            [](const EBFArrayBoxFactory& self) -> const Geometry&
            { return self.Geom(); },
            nb::rv_policy::reference_internal
        );

    // makeEBFabFactory: takes top of EB2::IndexSpace stack (must be built first)
    m.def(
        "make_eb_factory",
        [](const Geometry& geom,
           const BoxArray& ba,
           const DistributionMapping& dm,
           std::vector<int> ngrow_vec)
        {
            Vector<int> ngrow(ngrow_vec.begin(), ngrow_vec.end());
            // EBSupport::full gives volfrac, areafrac, centroid, normal
            return makeEBFabFactory(geom, ba, dm, ngrow, EBSupport::full);
        },
        nb::arg("geom"),
        nb::arg("ba"),
        nb::arg("dm"),
        nb::arg("ngrow") = std::vector<int>{4, 4, 2}
    );

    // For Python: Mesh.has_eb just tests `eb_factory is not None`. This
    // helper is provided for completeness when given an opaque factory.
    m.def(
        "is_eb_factory",
        [](nb::object f) -> bool
        {
            if (f.is_none()) return false;
            return nb::isinstance<EBFArrayBoxFactory>(f);
        },
        nb::arg("factory")
    );

    // --- EB MultiFab construction with the same single-chunk PaddedArena
    //     path the regular MultiFab binding uses. EBFArrayBoxFactory::create
    //     constructs each EBFArrayBox via info.arena(), so feeding it a
    //     PaddedArena bump-allocator yields a contiguous data buffer that
    //     contiguous_array() can view zero-copy. The per-fab EB metadata
    //     (cell-flag pointer) lives in the factory's separate FabArrays and
    //     does not interfere with data layout.
    m.def(
        "make_eb_multifab",
        [](const BoxArray& ba,
           const DistributionMapping& dm,
           int ncomp,
           int ngrow,
           const std::string& memory,
           int64_t padded_n_elems,
           const EBFArrayBoxFactory& factory)
        {
            // Compute required buffer size — same accounting as MultiFab ctor
            IntVect ng(ngrow);
            int64_t required = 0;
            for (int i = 0; i < ba.size(); ++i)
            {
                if (dm[i] == ParallelDescriptor::MyProc())
                {
                    Box grown = amrex::grow(ba[i], ng);
                    required += static_cast<int64_t>(grown.numPts()) * ncomp;
                }
            }

            int64_t padded = (padded_n_elems > 0 && padded_n_elems >= required)
                             ? padded_n_elems : required;

            Arena* base = neon::bindings::pickArena(memory);
            bool devAccess = base->isDeviceAccessible();
            bool hostAccess = base->isHostAccessible();

            MFInfo info;
            info.SetAllocSingleChunk(true);
            if (padded > required)
            {
                auto* parena = new neon::bindings::PaddedArena(
                    base,
                    static_cast<std::size_t>(required) * sizeof(Real),
                    static_cast<std::size_t>(padded) * sizeof(Real),
                    devAccess, hostAccess);
                info.SetArena(parena);
            }
            else
            {
                info.SetArena(base);
            }
            return MultiFab(ba, dm, ncomp, ngrow, info, factory);
        },
        nb::arg("ba"),
        nb::arg("dm"),
        nb::arg("ncomp"),
        nb::arg("ngrow"),
        nb::arg("memory") = "default",
        nb::arg("padded_n_elems") = 0,
        nb::arg("factory"),
        nb::keep_alive<0, 7>()
    );

    // --- EB utility functions ---
    m.def(
        "eb_set_covered",
        [](MultiFab& mf, Real val) { EB_set_covered(mf, val); },
        nb::arg("mf"),
        nb::arg("val") = Real(0.0)
    );

    m.def(
        "eb_set_covered",
        [](MultiFab& mf, int icomp, int ncomp, int ngrow, Real val)
        { EB_set_covered(mf, icomp, ncomp, ngrow, val); },
        nb::arg("mf"),
        nb::arg("icomp"),
        nb::arg("ncomp"),
        nb::arg("ngrow"),
        nb::arg("val")
    );

    m.def(
        "eb_set_covered_faces",
        [](MultiFab& fx, MultiFab& fy, MultiFab& fz, Real val)
        {
            Array<MultiFab*, AMREX_SPACEDIM> umac{AMREX_D_DECL(&fx, &fy, &fz)};
            EB_set_covered_faces(umac, val);
        },
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("val") = Real(0.0)
    );
#else
    (void)m;
#endif
}
