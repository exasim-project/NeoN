// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include <AMReX_Config.H>

#ifdef AMREX_USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF_Sphere.H>
#include <AMReX_EB2_IF_Cylinder.H>
#include <AMReX_EB2_IF_Plane.H>
#include <AMReX_EB2_IF_Box.H>
#include <AMReX_EB2_IF_AllRegular.H>
#include <AMReX_EB2_GeometryShop.H>
#include <AMReX_Geometry.H>
#endif

namespace nb = nanobind;

void registerEB2(nb::module_& m)
{
#ifdef AMREX_USE_EB
    using namespace amrex;

    // Each implicit function is bound as an opaque Python type. The
    // corresponding eb2_build_* helper instantiates EB2::GeometryShop<IF> and
    // calls EB2::Build, hiding the C++ template machinery from Python.

    nb::class_<EB2::SphereIF>(m, "EB2_SphereIF")
        .def(
            "__init__",
            [](EB2::SphereIF* self, Real radius, std::array<Real, 3> center, bool fluid_inside)
            {
                RealArray c{AMREX_D_DECL(center[0], center[1], center[2])};
                new (self) EB2::SphereIF(radius, c, fluid_inside);
            },
            nb::arg("radius"),
            nb::arg("center"),
            nb::arg("fluid_inside") = false
        );

    nb::class_<EB2::CylinderIF>(m, "EB2_CylinderIF")
        .def(
            "__init__",
            [](EB2::CylinderIF* self,
               Real radius,
               int direction,
               std::array<Real, 3> center,
               bool fluid_inside)
            {
                RealArray c{AMREX_D_DECL(center[0], center[1], center[2])};
                new (self) EB2::CylinderIF(radius, direction, c, fluid_inside);
            },
            nb::arg("radius"),
            nb::arg("direction"),
            nb::arg("center"),
            nb::arg("fluid_inside") = false
        )
        .def(
            "__init__",
            [](EB2::CylinderIF* self,
               Real radius,
               Real height,
               int direction,
               std::array<Real, 3> center,
               bool fluid_inside)
            {
                RealArray c{AMREX_D_DECL(center[0], center[1], center[2])};
                new (self) EB2::CylinderIF(radius, height, direction, c, fluid_inside);
            },
            nb::arg("radius"),
            nb::arg("height"),
            nb::arg("direction"),
            nb::arg("center"),
            nb::arg("fluid_inside") = false
        );

    nb::class_<EB2::PlaneIF>(m, "EB2_PlaneIF")
        .def(
            "__init__",
            [](EB2::PlaneIF* self,
               std::array<Real, 3> point,
               std::array<Real, 3> normal,
               bool fluid_inside)
            {
                RealArray p{AMREX_D_DECL(point[0], point[1], point[2])};
                RealArray n{AMREX_D_DECL(normal[0], normal[1], normal[2])};
                new (self) EB2::PlaneIF(p, n, fluid_inside);
            },
            nb::arg("point"),
            nb::arg("normal"),
            nb::arg("fluid_inside") = false
        );

    nb::class_<EB2::BoxIF>(m, "EB2_BoxIF")
        .def(
            "__init__",
            [](EB2::BoxIF* self,
               std::array<Real, 3> lo,
               std::array<Real, 3> hi,
               bool fluid_inside)
            {
                RealArray l{AMREX_D_DECL(lo[0], lo[1], lo[2])};
                RealArray h{AMREX_D_DECL(hi[0], hi[1], hi[2])};
                new (self) EB2::BoxIF(l, h, fluid_inside);
            },
            nb::arg("lo"),
            nb::arg("hi"),
            nb::arg("fluid_inside") = false
        );

    nb::class_<EB2::AllRegularIF>(m, "EB2_AllRegularIF")
        .def(nb::init<>());

    // --- eb2_build_<shape>: instantiate GeometryShop<IF> + EB2::Build ---
    //
    // Each helper builds a fresh IndexSpace and pushes it onto the EB2 stack;
    // EB2::IndexSpace::top() returns it. Subsequent EBFArrayBoxFactory
    // construction picks it up.
    auto build_with = [](auto if_obj,
                         const Geometry& geom,
                         int required_coarsening_level,
                         int max_coarsening_level)
    {
        EB2::Build(EB2::makeShop(if_obj),
                   geom,
                   required_coarsening_level,
                   max_coarsening_level);
    };

    m.def(
        "eb2_build_sphere",
        [build_with](const EB2::SphereIF& sph,
                     const Geometry& geom,
                     int required_coarsening_level,
                     int max_coarsening_level)
        { build_with(sph, geom, required_coarsening_level, max_coarsening_level); },
        nb::arg("sphere"),
        nb::arg("geom"),
        nb::arg("required_coarsening_level") = 0,
        nb::arg("max_coarsening_level") = 100
    );

    m.def(
        "eb2_build_cylinder",
        [build_with](const EB2::CylinderIF& cyl,
                     const Geometry& geom,
                     int required_coarsening_level,
                     int max_coarsening_level)
        { build_with(cyl, geom, required_coarsening_level, max_coarsening_level); },
        nb::arg("cylinder"),
        nb::arg("geom"),
        nb::arg("required_coarsening_level") = 0,
        nb::arg("max_coarsening_level") = 100
    );

    m.def(
        "eb2_build_plane",
        [build_with](const EB2::PlaneIF& pl,
                     const Geometry& geom,
                     int required_coarsening_level,
                     int max_coarsening_level)
        { build_with(pl, geom, required_coarsening_level, max_coarsening_level); },
        nb::arg("plane"),
        nb::arg("geom"),
        nb::arg("required_coarsening_level") = 0,
        nb::arg("max_coarsening_level") = 100
    );

    m.def(
        "eb2_build_box",
        [build_with](const EB2::BoxIF& bx,
                     const Geometry& geom,
                     int required_coarsening_level,
                     int max_coarsening_level)
        { build_with(bx, geom, required_coarsening_level, max_coarsening_level); },
        nb::arg("box"),
        nb::arg("geom"),
        nb::arg("required_coarsening_level") = 0,
        nb::arg("max_coarsening_level") = 100
    );

    m.def(
        "eb2_build_all_regular",
        [](const Geometry& geom,
           int required_coarsening_level,
           int max_coarsening_level)
        {
            EB2::Build(EB2::makeShop(EB2::AllRegularIF()),
                       geom,
                       required_coarsening_level,
                       max_coarsening_level);
        },
        nb::arg("geom"),
        nb::arg("required_coarsening_level") = 0,
        nb::arg("max_coarsening_level") = 100
    );

    // EB2::Build (via makeShop) calls IndexSpace::push internally. Provide
    // a way to clear the stack for tests / repeated runs.
    m.def("eb2_clear", []() { EB2::IndexSpace::clear(); });

    m.def("has_eb_support", []() { return true; });
#else
    m.def("has_eb_support", []() { return false; });
#endif
}
