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

// Bodies moved unchanged from solve/persistent.cpp; see precond.hpp for why.

namespace blockamr::la
{

namespace
{

// Refuse a precision the shipped GmgPrecondT hierarchy has no level type for.
void requireHierarchyPrecision(const GmgConfig& gmg)
{
    // bf16 is not a typo: it exists, but only for precond='gmg_kokkos'. The shipped GmgPrecondT
    // hierarchy is fp64/fp32 only.
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
}

// Refuse a knob the ported Kokkos V-cycle would have to ignore.
void requireKokkosVcycleKnobs(const GmgConfig& gmg)
{
    // Refused rather than ignored: accepting a knob that does nothing would report a Krylov
    // bottom in the configuration and run fixed sweeps. The ported V-cycle has no Ginkgo, so
    // GmgBottomOp cannot reach it.
    if (gmg.bottomSolver != "smoother")
    {
        throw std::runtime_error(
            "FaceCoeffSolver: precond='gmg_kokkos' has no Krylov bottom solve, so "
            "gmg_bottom_solver='"
            + gmg.bottomSolver
            + "' would silently run gmg_coarsest_sweeps sweeps. Use "
              "precond='gmg' for a Krylov bottom."
        );
    }
    // The Kokkos V-cycle assumes a self-adjoint cycle; no path in it honours symmetric=False.
    if (!gmg.symmetric)
    {
        throw std::runtime_error(
            "FaceCoeffSolver: precond='gmg_kokkos' assumes a symmetric operator; "
            "symmetric=False needs precond='gmg'"
        );
    }
    if (gmg.smoother != "rbgs")
    {
        throw std::runtime_error(
            "FaceCoeffSolver: precond='gmg_kokkos' has only the red-black smoother, not '"
            + gmg.smoother + "'"
        );
    }
}

// The Kokkos V-cycle options `config` asks for.
blockamr::KokkosGmgOpts kokkosVcycleOpts(const SolverConfig& config, const BcArray& bcArr)
{
    blockamr::KokkosGmgOpts opts;
    opts.cycles = config.precondCycles;
    opts.preSweeps = config.gmg.preSweeps;
    opts.postSweeps = config.gmg.postSweeps;
    opts.coarsestSweeps = config.gmg.coarsestSweeps;
    opts.maxLevels = config.gmg.maxLevels;
    opts.minBottom = config.gmg.minBottom;
    opts.omega = config.gmg.omega;
    // Straight through: makeKokkosGmgApply throws on an unknown spelling, so a typo cannot
    // quietly run fp64. This is the only precond that has a bf16 hierarchy.
    opts.precision = config.gmg.precision;
    // Likewise: it rejects an unknown spelling and a coefficient type wider than the fields.
    opts.coeffPrecision = config.gmg.coeffPrecision;
    // The parsed spec straight through: the same homogeneous reflection as precond="gmg",
    // built once per level as a device plan instead of a per-box AMReX launch.
    opts.bc = bcArr;
    opts.aggLevel0Size = config.gmg.aggLevel0Size;
    return opts;
}

} // namespace

GmgHierarchy buildGmgHierarchy(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const FaceCoeffLevel& level,
    const BcArray& bcArr,
    const SolverConfig& config
)
{
    const GmgConfig& gmg = config.gmg;
    requireHierarchyPrecision(gmg);
    auto makeGmg = [&](auto tag) -> GmgHierarchy
    {
        using T = decltype(tag);
        auto p = GmgPrecondT<T>::create(
            exec, n, level, bcArr, GmgPrecondSpec {config.precondCycles, gmg}
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
    // `alpha` for its LAYOUT only: the V-cycle's scratch fabs copy its ba/dm.
    return gko::share(
        MlmgPrecond::create(std::move(exec), config.precondMlmg, alpha, n, config.precondCycles)
    );
}

FaceCoeffPrecond makeFaceCoeffPrecond(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const FaceCoeffLevel& level,
    const BcArray& bcArr,
    const SolverConfig& config
)
{
    FaceCoeffPrecond out;
    if (config.precondKind == PrecondKind::gmg)
    {
        out.op = buildGmgHierarchy(exec, n, level, bcArr, config).op;
        return out;
    }
    if (config.precondKind == PrecondKind::gmg_kokkos)
    {
        // The same V-cycle as precond="gmg" under the optimised Kokkos launchers, as a separate
        // object so the shipped GmgPrecondT stays untouched and both can run in one process.
        requireKokkosVcycleKnobs(config.gmg);
        const blockamr::KokkosGmgOpts opts = kokkosVcycleOpts(config, bcArr);
        // Handed back as well as wrapped: solver="mpir" wraps the SAME hierarchy in an fp32 LinOp,
        // and building it twice would double the setup and the device memory.
        out.kokkosVcycle = std::shared_ptr<blockamr::KokkosGmgApply>(blockamr::makeKokkosGmgApply(
            level.mesh.geom,
            constView(level.alpha),
            constView(level.upper),
            constView(level.lower),
            opts
        ));
        out.op = gko::share(GmgKokkosPrecond::create(exec, n, out.kokkosVcycle));
        return out;
    }
    // precond="none"/"mlmg": parseSolverConfig rejected anything but those four kinds.
    // precond_mlmg alone implies "mlmg" (pre-existing behaviour).
    // No class name: this build serves FaceCoeffSolver AND la::Solver.
    if (config.precondKind == PrecondKind::mlmg && config.precondMlmg == nullptr)
    {
        throw std::runtime_error("precond='mlmg' requires precond_mlmg");
    }
    out.op = makeMlmgPrecond(exec, n, *level.alpha, config);
    return out;
}

} // namespace blockamr::la
