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

GmgHierarchy buildGmgHierarchy(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const CellFieldLevel& alpha,
    const FaceFieldLevel& upper,
    const FaceFieldLevel& lower,
    const MeshLevel& mesh,
    const BcArray& bcArr,
    int precondCycles,
    const GmgConfig& gmg
)
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
    auto makeGmg = [&](auto tag) -> GmgHierarchy
    {
        using T = decltype(tag);
        auto p = GmgPrecondT<T>::create(
            exec,
            mesh.ba,
            mesh.dm,
            mesh.geom,
            n,
            &(*alpha),
            &upper[0],
            &lower[0],
            &upper[1],
            &lower[1],
            &upper[2],
            &lower[2],
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
    const CellFieldLevel& alpha,
    const FaceFieldLevel& upper,
    const FaceFieldLevel& lower,
    const MeshLevel& mesh,
    const BcArray& bcArr,
    const SolverConfig& config
)
{
    FaceCoeffPrecond out;
    if (config.precondKind == PrecondKind::gmg)
    {
        out.op = buildGmgHierarchy(
                     exec, n, alpha, upper, lower, mesh, bcArr, config.precondCycles, config.gmg
        )
                     .op;
    }
    else if (config.precondKind == PrecondKind::gmg_kokkos)
    {
        // The same V-cycle as precond="gmg" under the optimised Kokkos launchers, as a separate
        // object so the shipped GmgPrecondT stays untouched and both can run in one process.

        // Refused rather than ignored: accepting a knob that does nothing would report a Krylov
        // bottom in the configuration and run fixed sweeps. The ported V-cycle has no Ginkgo, so
        // GmgBottomOp cannot reach it.
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
        // The Kokkos V-cycle assumes a self-adjoint cycle; no path in it honours symmetric=False.
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
        // Straight through: makeKokkosGmgApply throws on an unknown spelling, so a typo cannot
        // quietly run fp64. This is the only precond that has a bf16 hierarchy.
        opts.precision = config.gmg.precision;
        // Likewise: it rejects an unknown spelling and a coefficient type wider than the fields.
        opts.coeffPrecision = config.gmg.coeffPrecision;
        // The parsed spec straight through: the same homogeneous reflection as precond="gmg",
        // built once per level as a device plan instead of a per-box AMReX launch.
        opts.bc = bcArr;
        opts.aggLevel0Size = config.gmg.aggLevel0Size;
        // Handed back as well as wrapped: solver="mpir" wraps the SAME hierarchy in an fp32 LinOp,
        // and building it twice would double the setup and the device memory.
        out.kokkosVcycle = std::shared_ptr<blockamr::KokkosGmgApply>(blockamr::makeKokkosGmgApply(
            mesh.geom, *alpha, upper[0], lower[0], upper[1], lower[1], upper[2], lower[2], opts
        ));
        out.op = gko::share(GmgKokkosPrecond::create(exec, n, out.kokkosVcycle));
    }
    else
    {
        // precond="none"/"mlmg": parseSolverConfig rejected anything but those four kinds.
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
