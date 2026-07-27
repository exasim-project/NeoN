// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// nanobind surface for the Kokkos-vs-AMReX operator bench. Kept apart from the
// implementations: those compile in the separate blockamr_bench object library (see
// CMakeLists.txt -- not an RDC fence, the whole module builds non-RDC), so both
// sides agree on exactly one shared header.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <AMReX_MultiFab.H>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "bindings.hpp"

#include "../../blockAmrSolvers/bench/kokkos_bench.hpp"
#include "../../blockAmrSolvers/kokkos/runtime.hpp"

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
           amrex::MultiFab* fx_lo,
           amrex::MultiFab* fy_lo,
           amrex::MultiFab* fz_lo,
           int pre_sweeps,
           int post_sweeps,
           int coarsest_sweeps,
           int max_levels,
           int min_bottom,
           double omega,
           bool agglomerate,
           int agg_grid_size,
           const std::string& precision,
           const std::string& coeff_precision,
           bool share_coeffs,
           const std::vector<int>& bc,
           int agg_level0_size,
           int iters,
           int batches)
        {
            blockamr::bench::GmgArgs args;
            args.geom = &geom;
            args.rhs = &rhs;
            args.alpha = &alpha;
            // Symmetric operator by default: the upper and lower coefficient of a
            // direction are the same face field, as the persistent solvers are handed
            // it. Passing fx_lo/fy_lo/fz_lo gives the lower coefficients separately,
            // which is what makes share_coeffs testable -- equal-but-distinct fabs
            // must be detected as shareable, and genuinely different ones must not.
            args.ux = &fx;
            args.lx = (fx_lo != nullptr) ? fx_lo : &fx;
            args.uy = &fy;
            args.ly = (fy_lo != nullptr) ? fy_lo : &fy;
            args.uz = &fz;
            args.lz = (fz_lo != nullptr) ? fz_lo : &fz;
            args.preSweeps = pre_sweeps;
            args.postSweeps = post_sweeps;
            args.coarsestSweeps = coarsest_sweeps;
            args.maxLevels = max_levels;
            args.minBottom = min_bottom;
            args.omega = omega;
            args.agglomerate = agglomerate;
            args.aggGridSize = agg_grid_size;
            args.precision = precision;
            args.coeffPrecision = coeff_precision;
            args.shareCoeffs = share_coeffs;
            args.aggLevel0Size = agg_level0_size;
            // Integers, not the solver's bc strings: parseBc lives in the Ginkgo-only
            // half of the module and this binding is always built. Empty = periodic.
            if (!bc.empty())
            {
                if (bc.size() != 6)
                {
                    throw std::runtime_error("bench_gmg_vcycle: bc needs 6 entries "
                                             "(xlo, xhi, ylo, yhi, zlo, zhi)");
                }
                for (std::size_t i = 0; i < 6; ++i)
                {
                    args.bc[i] = bc[i];
                }
            }

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
            // What the hierarchy DID, not what was asked for: share_coeffs is only
            // honoured for a symmetric operator.
            d["shared_coeffs"] = r.sharedCoeffs;
            d["agg_level0"] = r.aggLevel0;
            return d;
        },
        nb::arg("backend"),
        nb::arg("geom"),
        nb::arg("rhs"),
        nb::arg("alpha"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("fx_lo").none() = nb::none(),
        nb::arg("fy_lo").none() = nb::none(),
        nb::arg("fz_lo").none() = nb::none(),
        nb::arg("pre_sweeps") = 2,
        nb::arg("post_sweeps") = 2,
        nb::arg("coarsest_sweeps") = 8,
        nb::arg("max_levels") = 0,
        nb::arg("min_bottom") = 2,
        nb::arg("omega") = 1.0,
        nb::arg("agglomerate") = false,
        nb::arg("agg_grid_size") = 32,
        // The level storage type: "fp64", "fp32" or "bf16" -- kokkos_opt only.
        nb::arg("precision") = "fp64",
        // The COEFFICIENT storage type; "" (the default) means the same as
        // `precision`. May not be wider than it. kokkos_opt only.
        nb::arg("coeff_precision") = "",
        nb::arg("share_coeffs") = false,
        // Per side (xlo, xhi, ylo, yhi, zlo, zhi): 0 periodic, 1 homogeneous
        // Dirichlet, 2 homogeneous Neumann. Empty (the default) means all periodic.
        nb::arg("bc") = std::vector<int> {},
        // Target box size for level 0's own decomposition; 0 keeps the caller's boxes.
        nb::arg("agg_level0_size") = 0,
        nb::arg("iters") = 10,
        nb::arg("batches") = 5
    );
}
