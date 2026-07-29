// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/linearAlgebra/precond.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "NeoN/blockAmr/linearAlgebra/gmg/gmgPrecond.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/precond.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/mlmgOps.hpp"

// The bodies moved here from solve/persistent.cpp. See precond.hpp for why they
// moved and for the promise that they moved UNCHANGED: every line below was
// already running, at the same point in the same order, before this file existed.
// The only edits are mechanical -- a lambda capture becomes a parameter, `exec_`
// and `n_` become `exec` and `n`, and the two file-local blocks became one
// function each.

namespace blockamr::la
{

GmgHierarchy buildGmgHierarchy(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    const amrex::Geometry& geom,
    const BcArray& bcArr,
    int precondCycles,
    const GmgConfig& gmg
)
{
    // bf16 is named separately from an outright typo: it exists, but only for
    // precond='gmg_kokkos'. The shipped GmgPrecondT hierarchy is fp64/fp32, and
    // instantiating it for a storage-only type would mean porting its Chebyshev
    // smoother and lambda-max power iteration too.
    if (gmg.precision == "bf16")
    {
        throw std::runtime_error("FaceCoeffSolver: gmg_precision='bf16' needs precond='gmg_kokkos' "
                                 "(the shipped GMG hierarchy is fp64/fp32 only)");
    }
    if (gmg.precision != "fp64" && gmg.precision != "fp32")
    {
        throw std::runtime_error(
            "FaceCoeffSolver: unknown gmg_precision '" + gmg.precision
            + "' (expected 'fp64' or 'fp32')"
        );
    }
    auto makeGmg = [&](auto tag) -> GmgHierarchy
    {
        using T = decltype(tag);
        auto p = GmgPrecondT<T>::create(
            exec,
            alpha->boxArray(),
            alpha->DistributionMap(),
            geom,
            n,
            alpha,
            ux,
            lx,
            uy,
            ly,
            uz,
            lz,
            bcArr,
            precondCycles,
            gmg.preSweeps,
            gmg.postSweeps,
            gmg.coarsestSweeps,
            gmg.maxLevels,
            gmg.minBottom,
            gmg.smoother,
            gmg.omega,
            gmg.symmetric,
            gmg.bottomSolver,
            gmg.bottomMaxIter,
            gmg.bottomRtol
        );
        GmgHierarchy h;
        h.mf = p.get(); // GmgPrecondT<T>* -> const GmgApplyMf* (kept alive by h.op below)
        h.op = gko::share(std::move(p));
        return h;
    };
    return (gmg.precision == "fp32") ? makeGmg(float {}) : makeGmg(double {});
}

std::shared_ptr<const gko::LinOp> makeMlmgPrecond(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const amrex::MultiFab& alpha,
    const SolverConfig& config
)
{
    if (config.precondMlmg == nullptr)
    {
        return nullptr;
    }
    return gko::share(MlmgPrecond::create(
        std::move(exec),
        config.precondMlmg,
        alpha.boxArray(),
        alpha.DistributionMap(),
        n,
        config.precondCycles
    ));
}

FaceCoeffPrecond makeFaceCoeffPrecond(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    const amrex::Geometry& geom,
    const BcArray& bcArr,
    const SolverConfig& config
)
{
    FaceCoeffPrecond out;
    if (config.precondKind == PrecondKind::gmg)
    {
        out.op = buildGmgHierarchy(
                     exec,
                     n,
                     alpha,
                     ux,
                     lx,
                     uy,
                     ly,
                     uz,
                     lz,
                     geom,
                     bcArr,
                     config.precondCycles,
                     config.gmg
        )
                     .op;
    }
    else if (config.precondKind == PrecondKind::gmg_kokkos)
    {
        // The same V-cycle as precond="gmg", under the optimised Kokkos launchers
        // (gmgKokkos/apply.hpp). A separate object rather than a mode of GmgPrecondT:
        // that one is the shipped baseline and stays untouched, so both can run in
        // one process and be compared directly.
        // Refused rather than ignored, for the same reason every other
        // capability gap on this path is: accepting a knob that does nothing
        // reports a Krylov bottom in the configuration and runs fixed sweeps.
        // The ported V-cycle lives behind the bench fence and has no Ginkgo, so
        // GmgBottomOp cannot reach it; closing this means porting the bottom
        // solve to that side, not relaxing the check.
        if (config.gmg.bottomSolver != "smoother")
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' has no Krylov bottom solve, so "
                "gmg_bottom_solver='"
                + config.gmg.bottomSolver
                + "' would silently run gmg_coarsest_sweeps sweeps. Use "
                  "precond='gmg' for a Krylov bottom."
            );
        }
        // The Kokkos V-cycle carries the same symmetry assumptions the shipped one
        // does (an over-relaxed red-black sweep, a self-adjoint cycle), and has no
        // path that would honour symmetric=False.
        if (!config.gmg.symmetric)
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' assumes a symmetric operator; "
                "symmetric=False needs precond='gmg'"
            );
        }
        if (config.gmg.smoother != "rbgs")
        {
            throw std::runtime_error(
                "FaceCoeffSolver: precond='gmg_kokkos' has only the red-black smoother, not '"
                + config.gmg.smoother + "'"
            );
        }
        blockamr::KokkosGmgOpts opts;
        opts.cycles = config.precondCycles;
        opts.preSweeps = config.gmg.preSweeps;
        opts.postSweeps = config.gmg.postSweeps;
        opts.coarsestSweeps = config.gmg.coarsestSweeps;
        opts.maxLevels = config.gmg.maxLevels;
        opts.minBottom = config.gmg.minBottom;
        opts.omega = config.gmg.omega;
        // Straight through, unvalidated here: makeKokkosGmgApply parses it and
        // throws on an unknown spelling, so a typo cannot quietly run fp64. This
        // is the only precond that has a bf16 hierarchy.
        opts.precision = config.gmg.precision;
        // Likewise unvalidated here beyond the guard above: makeKokkosGmgApply
        // rejects an unknown spelling and a coefficient type wider than the fields.
        opts.coeffPrecision = config.gmg.coeffPrecision;
        // The parsed spec straight through: the ported V-cycle carries the same
        // homogeneous Dirichlet/Neumann reflection as precond="gmg", built once per
        // level as a device plan rather than as a per-box AMReX launch.
        opts.bc = bcArr;
        opts.aggLevel0Size = config.gmg.aggLevel0Size;
        // Handed back as well as wrapped: solver="mpir" wraps the SAME hierarchy in an
        // fp32 LinOp, and building it twice would double the setup and the device memory
        // for two views of one V-cycle.
        out.kokkosVcycle = std::shared_ptr<blockamr::KokkosGmgApply>(
            blockamr::makeKokkosGmgApply(geom, *alpha, *ux, *lx, *uy, *ly, *uz, *lz, opts)
        );
        out.op = gko::share(GmgKokkosPrecond::create(exec, n, out.kokkosVcycle));
    }
    else
    {
        // config.precondKind is one of {none, mlmg, gmg, gmg_kokkos}
        // (parseSolverConfig already rejected anything else), and gmg/
        // gmg_kokkos are handled by the two branches above, so this is
        // precond="none"/"mlmg".
        // precond_mlmg alone implies "mlmg" (pre-existing behaviour).
        if (config.precondKind == PrecondKind::mlmg && config.precondMlmg == nullptr)
        {
            throw std::runtime_error("FaceCoeffSolver: precond='mlmg' requires precond_mlmg");
        }
        out.op = makeMlmgPrecond(exec, n, *alpha, config);
    }
    return out;
}

} // namespace blockamr::la
