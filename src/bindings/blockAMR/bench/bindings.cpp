// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// nanobind surface for the Kokkos-vs-AMReX operator bench. Kept apart from the
// implementations: those compile in a non-RDC object library that must not see
// nanobind, so that both sides agree on exactly one shared header.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <AMReX_MultiFab.H>

#include "../bindings.hpp"
#include "kokkos_bench.hpp"

namespace nb = nanobind;

namespace
{

blockamr::bench::OpArgs makeArgs(
    amrex::MultiFab& out,
    amrex::MultiFab& in,
    amrex::MultiFab* fx,
    amrex::MultiFab* fy,
    amrex::MultiFab* fz,
    double a,
    double dx,
    double dy,
    double dz
)
{
    blockamr::bench::OpArgs args;
    args.out = &out;
    args.in = &in;
    args.fx = fx;
    args.fy = fy;
    args.fz = fz;
    args.a = a;
    args.dx = dx;
    args.dy = dy;
    args.dz = dz;
    return args;
}

} // namespace

void registerKokkosBench(nb::module_& m)
{
    m.def("kokkos_available", &blockamr::bench::kokkosInitialized);
    m.def("kokkos_execution_space", &blockamr::bench::kokkosExecutionSpace);
    m.def("kokkos_selftest", &blockamr::bench::kokkosSelftest, nb::arg("n") = 1024);
    m.def("kokkos_mf_sum", &blockamr::bench::kokkosMfSum, nb::arg("mf"));

    m.def("bench_operators", &blockamr::bench::benchOperators);

    m.def(
        "bench_operator_info",
        [](const std::string& name)
        {
            const auto info = blockamr::bench::benchOperatorInfo(name);
            nb::dict d;
            d["nghost"] = info.nghost;
            d["needs_faces"] = info.needsFaces;
            d["bytes_per_cell"] = info.bytesPerCell;
            return d;
        },
        nb::arg("name")
    );

    m.def(
        "apply_operator",
        [](const std::string& name,
           amrex::MultiFab& out,
           amrex::MultiFab& in,
           amrex::MultiFab* fx,
           amrex::MultiFab* fy,
           amrex::MultiFab* fz,
           double a,
           double dx,
           double dy,
           double dz)
        { blockamr::bench::applyOperator(name, makeArgs(out, in, fx, fy, fz, a, dx, dy, dz)); },
        nb::arg("name"),
        nb::arg("out"),
        nb::arg("in_"),
        nb::arg("fx").none() = nb::none(),
        nb::arg("fy").none() = nb::none(),
        nb::arg("fz").none() = nb::none(),
        nb::arg("a") = 1.0,
        nb::arg("dx") = 1.0,
        nb::arg("dy") = 1.0,
        nb::arg("dz") = 1.0
    );

    m.def(
        "bench_operator",
        [](const std::string& name,
           amrex::MultiFab& out,
           amrex::MultiFab& in,
           amrex::MultiFab* fx,
           amrex::MultiFab* fy,
           amrex::MultiFab* fz,
           double a,
           double dx,
           double dy,
           double dz,
           int iters,
           int batches)
        {
            const auto r = blockamr::bench::benchOperator(
                name, makeArgs(out, in, fx, fy, fz, a, dx, dy, dz), iters, batches
            );
            nb::dict d;
            d["ms_min"] = r.msMin;
            d["ms_median"] = r.msMedian;
            d["ms_enqueue"] = r.msEnqueue;
            d["gb_per_s"] = r.gbPerSec;
            d["ncells"] = r.ncells;
            d["nboxes"] = r.nboxes;
            return d;
        },
        nb::arg("name"),
        nb::arg("out"),
        nb::arg("in_"),
        nb::arg("fx").none() = nb::none(),
        nb::arg("fy").none() = nb::none(),
        nb::arg("fz").none() = nb::none(),
        nb::arg("a") = 1.0,
        nb::arg("dx") = 1.0,
        nb::arg("dy") = 1.0,
        nb::arg("dz") = 1.0,
        nb::arg("iters") = 50,
        nb::arg("batches") = 5
    );

    m.def("bench_gmg_backends", &blockamr::bench::benchGmgBackends);

    m.def(
        "bench_gmg_vcycle",
        [](const std::string& backend,
           const amrex::Geometry& geom,
           amrex::MultiFab& rhs,
           amrex::MultiFab& alpha,
           amrex::MultiFab& fx,
           amrex::MultiFab& fy,
           amrex::MultiFab& fz,
           int pre_sweeps,
           int post_sweeps,
           int coarsest_sweeps,
           int max_levels,
           int min_bottom,
           double omega,
           int iters,
           int batches)
        {
            blockamr::bench::GmgArgs args;
            args.geom = &geom;
            args.rhs = &rhs;
            args.alpha = &alpha;
            // Symmetric operator: the upper and lower coefficient of a direction are
            // the same face field, as the persistent solvers are handed it.
            args.ux = &fx;
            args.lx = &fx;
            args.uy = &fy;
            args.ly = &fy;
            args.uz = &fz;
            args.lz = &fz;
            args.preSweeps = pre_sweeps;
            args.postSweeps = post_sweeps;
            args.coarsestSweeps = coarsest_sweeps;
            args.maxLevels = max_levels;
            args.minBottom = min_bottom;
            args.omega = omega;

            const auto r = blockamr::bench::benchGmgVcycle(backend, args, iters, batches);
            nb::dict d;
            d["ms_min"] = r.msMin;
            d["ms_median"] = r.msMedian;
            d["ms_enqueue"] = r.msEnqueue;
            d["nlevels"] = r.nlevels;
            d["boxes_per_level"] = r.boxesPerLevel;
            d["cells_per_level"] = r.cellsPerLevel;
            d["resid0"] = r.resid0;
            d["resid1"] = r.resid1;
            return d;
        },
        nb::arg("backend"),
        nb::arg("geom"),
        nb::arg("rhs"),
        nb::arg("alpha"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("pre_sweeps") = 2,
        nb::arg("post_sweeps") = 2,
        nb::arg("coarsest_sweeps") = 8,
        nb::arg("max_levels") = 0,
        nb::arg("min_bottom") = 2,
        nb::arg("omega") = 1.0,
        nb::arg("iters") = 10,
        nb::arg("batches") = 5
    );
}
