// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The production Kokkos GMG V-cycle: Vcycle<Backend,T,TC> plus LevelT, sameField,
// KokkosOptGmgBackend and the Precision/PrecPair enums. Split out of the old single
// bench TU so the shipped precond="gmg_kokkos" path (apply.cpp) does not drag the
// four-backend benchmark harness into the wheel; see bench/gmgVcycleBench.cpp.
//
// A header because Vcycle is a template: apply.cpp instantiates it for
// KokkosOptGmgBackend only, bench/gmgVcycleBench.cpp for all four backends.

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include <AMReX_MultiFab.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>

#include "NeoN/blockAmr/bench/kokkosBench.hpp"
#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/halo.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/kernels.hpp"

namespace blockamr
{

namespace
{

// The level scalar is a template parameter, as in production's GmgPrecondT: the
// hierarchy below level 0 can run narrower while the operator and the caller's fields
// stay fp64, shrinking the traffic of a bandwidth-bound smoother. Only kokkos_opt is
// instantiated narrow; the baselines stay fp64, which is what keeps them baselines.
// bf16 is STORAGE only (la::Bf16 converts on read, rounds on write, kernels compute in
// la::GmgComputeT<T>) -- see gmg/bf16.hpp.
enum class Precision
{
    fp64,
    fp32,
    bf16
};

Precision parsePrecision(const std::string& p)
{
    if (p == "fp64")
    {
        return Precision::fp64;
    }
    if (p == "fp32")
    {
        return Precision::fp32;
    }
    if (p == "bf16")
    {
        return Precision::bf16;
    }
    throw std::runtime_error(
        "benchGmgVcycle: unknown precision '" + p + "' (expected 'fp64', 'fp32' or 'bf16')"
    );
}

// Only used to reject a coefficient type WIDER than the field type: paying traffic for
// accuracy in the array that needs it least is a configuration mistake, not a
// trade-off.
int precisionBytes(Precision p)
{
    switch (p)
    {
    case Precision::fp64:
        return 8;
    case Precision::fp32:
        return 4;
    case Precision::bf16:
        return 2;
    }
    return 8;
}

// The coefficient precision defaults to the field precision.
Precision parseCoeffPrecision(const std::string& coeff, const std::string& field)
{
    const Precision f = parsePrecision(field);
    if (coeff.empty())
    {
        return f;
    }
    const Precision c = parsePrecision(coeff);
    if (precisionBytes(c) > precisionBytes(f))
    {
        throw std::runtime_error(
            "benchGmgVcycle: coeff_precision '" + coeff + "' is wider than precision '" + field
            + "'; narrow the fields first"
        );
    }
    return c;
}

// The fused kernels with every AMReX operation lifted out of the timed cycle (halo.hpp
// supplies the halo exchange, the zero fill and the agglomeration transfers), so the
// per-kernel fence goes too: consecutive Kokkos kernels on one execution space are
// already stream-ordered, and the host can now run a coarse level's launches ahead of
// the arithmetic above it. The fence could not be dropped earlier -- the FillBoundary
// after each colour sweep needed exactly the ordering it gave.
//
// ONE RANK ONLY, a limit of the plans and not of the kernels: a CopyPlan task names two
// LOCAL box indices, so a ghost owned by another rank has no device address (halo.hpp).
// On >1 rank the CELL KERNELS stay Kokkos and the DATA MOVEMENTS go back to AMReX --
// what `kokkos_fused` already does. Decided in `Vcycle::amrexFree_`.
struct KokkosOptGmgBackend
{
    static constexpr const char* tag = "kokkos_opt";
    static constexpr bool amrexFreeCycle = true;
    static constexpr bool canShareCoeffs = true;

    // The reset/residual path still crosses into AMReX, outside the timed region.
    static void beforeAmrexRead() { Kokkos::fence(); }

    static void afterAmrexWrite() { amrex::Gpu::streamSynchronizeAll(); }

    // The level scalar is deduced from the fabs, so this backend alone can run narrow.
    template<class... A>
    static void gsColor(A&&... a)
    {
        gmgGsColorKokkosFused(std::forward<A>(a)..., /*fence=*/false);
    }

    template<class... A>
    static void residRestrict(A&&... a)
    {
        gmgResidRestrictKokkosFused(std::forward<A>(a)..., /*fence=*/false);
    }

    template<class... A>
    static void prolongAdd(A&&... a)
    {
        gmgProlongAddKokkosFused(std::forward<A>(a)..., /*fence=*/false);
    }
};

// Are these two fields the same numbers everywhere? ux == lx pointwise IS symmetry of
// the operator (cell i's east coefficient and cell i+1's west one are the entries
// A[i][i+1] and A[i+1][i], both stored at face i+1), and so the exact condition for one
// array to stand in for both. Bitwise on purpose: a tolerance would silently symmetrise
// a near-symmetric operator. The common case (the same fab passed twice) is settled by
// the pointer test; setup only, so the reduction is never timed.
bool sameField(const amrex::MultiFab& a, const amrex::MultiFab& b)
{
    if (&a == &b)
    {
        return true;
    }
    if (a.boxArray() != b.boxArray() || a.DistributionMap() != b.DistributionMap())
    {
        return false;
    }
    const auto aa = a.const_arrays();
    const auto bb = b.const_arrays();
    double diff = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpMax> {},
        amrex::TypeList<double> {},
        a,
        amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) -> amrex::GpuTuple<double>
        { return {amrex::Math::abs(aa[box](i, j, k) - bb[box](i, j, k))}; }
    );
    // ParReduce is this rank's boxes only, and the answer decides how many face fabs a
    // level keeps: ranks that disagreed would build different hierarchies.
    amrex::ParallelAllReduce::Max(diff, amrex::ParallelContext::CommunicatorSub());
    return diff == 0.0;
}

// One multigrid level, as in GmgLevelT: geometry, rediscretised coefficients and
// preallocated work fields (sol needs 1 ghost for the stencil, rhs is valid-only).
template<class T, class TC>
struct LevelT
{
    // Production's fab type (FabArray<BaseFab<T>>), so the production kernels apply
    // unchanged. Fields carry T, coefficients TC; see GmgGsCell for why they split.
    using Fab = la::GmgFab<T>;
    using CoeffFab = la::GmgFab<TC>;

    amrex::Geometry geom;
    std::unique_ptr<CoeffFab> alpha, ux, lx, uy, ly, uz, lz;
    std::unique_ptr<Fab> sol, rhs;

    // lx/ly/lz are NULL on a level sharing one face coefficient per direction
    // (GmgArgs::shareCoeffs, kokkos_opt only): for a symmetric operator ux and lx hold
    // identical numbers, so passing ux for both is the same arithmetic on half the
    // storage. Always read the lower coefficients through these accessors, so a shared
    // level cannot be dereferenced by accident.
    [[nodiscard]] const CoeffFab& lxf() const { return lx ? *lx : *ux; }
    [[nodiscard]] const CoeffFab& lyf() const { return ly ? *ly : *uy; }
    [[nodiscard]] const CoeffFab& lzf() const { return lz ? *lz : *uz; }
    [[nodiscard]] bool shared() const { return lx == nullptr; }

    // Agglomerated levels only. This level's BoxArray is a fresh decomposition, so it
    // no longer matches the fine level box for box and the inter-level kernels -- which
    // address fine and coarse at the SAME local box index -- cannot reach it. These
    // hold the restriction output and the prolongation input on coarsen(fine BoxArray,
    // 2) with the FINE DistributionMapping; ParallelCopy bridges to this level's own
    // fields.
    std::unique_ptr<Fab> xferRhs, xferSol;
    bool agglomerated = false;

    // kokkos_opt only, empty otherwise: this level's data movements resolved to device
    // tables at setup -- sol's ghost exchange, the homogeneous domain-boundary
    // reflection (empty on a periodic mesh), and the two directions of the
    // agglomeration transfer.
    CopyPlan halo, bc, xferIn, xferOut;
};

template<class Backend, class T, class TC = T>
class Vcycle
{
public:

    using Level = LevelT<T, TC>;
    using Fab = la::GmgFab<T>;
    using CoeffFab = la::GmgFab<TC>;

    Vcycle(const GmgArgs& args)
        : preSweeps_(args.preSweeps), postSweeps_(args.postSweeps),
          coarsestSweeps_(args.coarsestSweeps), omega_(args.omega), bc_(args.bc),
          hasPhysBc_(std::any_of(args.bc.begin(), args.bc.end(), [](int b) { return b != 0; })),
          amrexFree_(Backend::amrexFreeCycle && amrex::ParallelContext::NProcsSub() == 1)
    {
        const amrex::BoxArray& ba = args.alpha->boxArray();
        const amrex::DistributionMapping& dm = args.alpha->DistributionMap();

        // One face coefficient per direction instead of an upper/lower pair -- only
        // when asked, only if the backend allows it, and only if the operator really is
        // symmetric: one that quietly lost its lower coefficients would solve a
        // different system at full speed.
        shared_ = Backend::canShareCoeffs && args.shareCoeffs && sameField(*args.ux, *args.lx)
               && sameField(*args.uy, *args.ly) && sameField(*args.uz, *args.lz);

        // Level 0 on ITS OWN decomposition, when asked. Every other level picks its
        // boxes freely; level 0 is the one the caller addresses, so bigger boxes cost
        // an interface (a caller-layout fab at each end of an apply plus a plan-driven
        // copy). What it buys is halo traffic: 32^3 boxes carry 19% ghost overhead
        // against 9.4% for 64^3, and level 0 is 7/8 of the cells in the hierarchy.
        amrex::BoxArray l0ba = ba;
        amrex::DistributionMapping l0dm = dm;
        // Only available where the plans are. On >1 rank the caller's layout is kept:
        // the trade was measured against a local copy, not against an MPI ParallelCopy.
        if (amrexFree_)
        {
            if (args.aggLevel0Size > 0)
            {
                amrex::BoxArray tba(args.geom->Domain());
                tba.maxSize(args.aggLevel0Size);
                // Strictly fewer boxes, or the interface is pure cost.
                if (tba.size() < ba.size())
                {
                    l0ba = tba;
                    l0dm = amrex::DistributionMapping(tba);
                    aggL0_ = true;
                }
            }
        }

        levels_.push_back(makeLevel(l0ba, l0dm, *args.geom, shared_));
        if (aggL0_)
        {
            // The convert-copies need matching layouts, so rediscretise on the caller's
            // decomposition and ParallelCopy across. Setup, so the temporary is
            // untimed.
            Level t = makeLevel(ba, dm, *args.geom, shared_);
            copyCallerCoeffs(args, t);
            copyCoeffs(t, levels_[0]);
            iface_ = makeMf(ba, dm, 0);
        }
        else
        {
            copyCallerCoeffs(args, levels_[0]);
        }

        // Coarsen while the BoxArray stays coarsenable and the coarse domain keeps
        // >= minBottom cells per direction. Without agglomeration the fine BoxArray is
        // coarsened in place and the DistributionMapping reused, so only the box SIZE
        // shrinks — production behaviour, and why the coarsest level is launch-bound.
        while (true)
        {
            if (args.maxLevels > 0 && static_cast<int>(levels_.size()) >= args.maxLevels)
            {
                break;
            }
            // Copies, not references: push_back below moves the Level structs.
            const amrex::BoxArray fba = levels_.back().alpha->boxArray();
            const amrex::DistributionMapping fdm = levels_.back().alpha->DistributionMap();
            const amrex::Geometry fgeom = levels_.back().geom;
            if (!fba.coarsenable(2, 2))
            {
                break;
            }
            const amrex::Box cdom = amrex::coarsen(fgeom.Domain(), 2);
            if (cdom.shortside() < args.minBottom)
            {
                break;
            }
            amrex::BoxArray cba = fba;
            cba.coarsen(2);
            // Each level carries its OWN geometry: the stencil and the rediscretised
            // coefficients belong to this level's dx, never to level 0's.
            const amrex::Geometry cgeom(
                cdom,
                fgeom.ProbDomain(),
                fgeom.Coord(),
                {fgeom.isPeriodic(0), fgeom.isPeriodic(1), fgeom.isPeriodic(2)}
            );

            // Agglomeration: a fresh aggGridSize-capped decomposition of the coarse
            // domain, taken only when it has strictly fewer boxes than coarsening in
            // place. Same mechanism as MLMG's LPInfo::do_agglomeration but a different
            // trigger: MLMG reduces the number of MPI ranks with work (default
            // agg_grid_size 8 in 3D, which on one GPU would leave the box count
            // untouched), while the cost here is per-box launches -- so the trigger is
            // the box count itself.
            amrex::BoxArray aba = cba;
            amrex::DistributionMapping adm = fdm;
            bool agg = false;
            if (args.agglomerate)
            {
                amrex::BoxArray tba(cdom);
                tba.maxSize(args.aggGridSize);
                if (tba.size() < cba.size())
                {
                    aba = tba;
                    adm = amrex::DistributionMapping(tba);
                    agg = true;
                }
            }

            levels_.push_back(makeLevel(aba, adm, cgeom, shared_));
            const Level& fl = levels_[levels_.size() - 2];
            Level& c = levels_.back();
            if (!agg)
            {
                restrictCoeffs(fl, c);
            }
            else
            {
                c.agglomerated = true;
                c.xferRhs = makeMf(cba, fdm, 0);
                c.xferSol = makeMf(cba, fdm, 0);
                // The restriction kernels only speak the fine level's layout, so
                // rediscretise there and copy onto this level's decomposition
                // (untimed).
                Level t = makeLevel(cba, fdm, cgeom, shared_);
                restrictCoeffs(fl, t);
                copyCoeffs(t, c);
            }
        }

        // Resolve the data movements once the hierarchy is final; the timed cycle then
        // sees a device table and one launch.
        if (amrexFree_)
        {
            if (aggL0_)
            {
                ifaceIn_ = makeCopyPlan(*levels_[0].rhs, *iface_);
                ifaceOut_ = makeCopyPlan(*iface_, *levels_[0].sol);
            }
            for (Level& L : levels_)
            {
                L.halo = makeHaloPlan(*L.sol, L.geom.periodicity());
                if (hasPhysBc_)
                {
                    // Coarse domains are the fine one coarsened, so every level has the
                    // same physical faces and one bc spec applies throughout.
                    L.bc = makeBcPlan(*L.sol, L.geom.Domain(), bc_);
                }
                if (L.agglomerated)
                {
                    L.xferIn = makeCopyPlan(*L.rhs, *L.xferRhs);
                    L.xferOut = makeCopyPlan(*L.xferSol, *L.sol);
                }
            }
        }
        amrex::Gpu::streamSynchronize();
    }

    // L0 rhs := the caller's rhs, L0 sol := 0 — the state a preconditioner apply
    // starts from (z0 = 0, so this applies M^{-1} rather than warm-starting).
    void reset(const amrex::MultiFab& rhs)
    {
        if (aggL0_)
        {
            la::gmgConvertCopy(*iface_, rhs, /*onDevice=*/true);
            Backend::afterAmrexWrite();
            gmgCopyKokkos<T>(*levels_[0].rhs, *iface_, ifaceIn_);
            gmgZeroKokkos<T>(*levels_[0].sol);
            Kokkos::fence();
            return;
        }
        Backend::beforeAmrexRead(); // a previous cycle's kernels may still be in flight
        la::gmgConvertCopy(*levels_[0].rhs, rhs, /*onDevice=*/true);
        levels_[0].sol->setVal(T(0));
        amrex::Gpu::streamSynchronize();
    }

    void cycles(int n)
    {
        for (int c = 0; c < n; ++c)
        {
            vcycle(0);
        }
    }

    // Preconditioner apply on flat device vectors: L0 rhs <- r, L0 sol <- 0, n
    // V-cycles, z <- L0 sol. Same sequence as GmgPrecondT::apply_impl, including the
    // two AMReX transfers, so a solver-level comparison measures the cycle and not the
    // plumbing.
    template<class V>
    void applyFlat(const V* r, V* z, int nCycles)
    {
        Level& L0 = levels_.front();
        // The flat vector's cell order is the CALLER's, so with an agglomerated level 0
        // the scatter lands in the staging fab and a plan copy carries it across, and
        // back on the way out -- once per APPLY, so precond_cycles > 1 amortises it.
        if (aggL0_)
        {
            la::scatter_device(r, *iface_);
            Backend::afterAmrexWrite();
            gmgCopyKokkos<T>(*L0.rhs, *iface_, ifaceIn_);
            gmgZeroKokkos<T>(*L0.sol); // z0 = 0: apply M^{-1}, not a warm-started solve
            cycles(nCycles);
            gmgCopyKokkos<T>(*iface_, *L0.sol, ifaceOut_);
            Backend::beforeAmrexRead();
            la::gather_device(*iface_, z, 1.0);
            amrex::Gpu::streamSynchronize();
            return;
        }
        Backend::beforeAmrexRead(); // a previous apply's kernels may still be in flight
        la::scatter_device(r, *L0.rhs);
        L0.sol->setVal(T(0)); // z0 = 0: apply M^{-1}, not a warm-started solve
        Backend::afterAmrexWrite();
        cycles(nCycles);
        Backend::beforeAmrexRead();
        la::gather_device(*L0.sol, z, 1.0);
        amrex::Gpu::streamSynchronize(); // z complete before the caller reads it
    }

    // sum((rhs - A sol)^2) on the finest level. Reporting only -- the gate that says
    // the timed cycle did the work -- so it stays an AMReX reduction and is never
    // timed.
    double residSumSq()
    {
        Level& L = levels_.front();
        fillGhosts(L);
        Backend::beforeAmrexRead();
        const auto psi = L.sol->const_arrays();
        const auto b = L.rhs->const_arrays();
        const auto ax = L.ux->const_arrays();
        const auto lxa = L.lxf().const_arrays();
        const auto ay = L.uy->const_arrays();
        const auto lya = L.lyf().const_arrays();
        const auto az = L.uz->const_arrays();
        const auto lza = L.lzf().const_arrays();
        const auto al = L.alpha->const_arrays();
        double sum = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpSum> {},
            amrex::TypeList<double> {},
            *L.rhs,
            amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) -> amrex::GpuTuple<double>
            {
                const double aE = ax[box](i + 1, j, k);
                const double aW = lxa[box](i, j, k);
                const double aN = ay[box](i, j + 1, k);
                const double aS = lya[box](i, j, k);
                const double aT = az[box](i, j, k + 1);
                const double aB = lza[box](i, j, k);
                const double off = aE * psi[box](i + 1, j, k) + aW * psi[box](i - 1, j, k)
                                 + aN * psi[box](i, j + 1, k) + aS * psi[box](i, j - 1, k)
                                 + aT * psi[box](i, j, k + 1) + aB * psi[box](i, j, k - 1);
                const double diag = al[box](i, j, k) - (aE + aW + aN + aS + aT + aB);
                const double r = b[box](i, j, k) - (diag * psi[box](i, j, k) + off);
                return {r * r};
            }
        );
        // This rank's boxes only, like every ParReduce; the caller prints its root as
        // THE residual of the cycle.
        amrex::ParallelAllReduce::Sum(sum, amrex::ParallelContext::CommunicatorSub());
        return sum;
    }

    int nlevels() const { return static_cast<int>(levels_.size()); }

    // What the hierarchy actually does, not what was requested (see sameField).
    bool sharedCoeffs() const { return shared_; }

    // Whether level 0 really got its own decomposition; asking does not guarantee it.
    bool aggLevel0() const { return aggL0_; }

    // Boxes and cells PER LEVEL: the point is that the box count does not shrink while
    // the cell count does.
    std::vector<int> boxesPerLevel() const
    {
        std::vector<int> v;
        for (const Level& L : levels_)
        {
            v.push_back(static_cast<int>(L.alpha->boxArray().size()));
        }
        return v;
    }

    std::vector<long> cellsPerLevel() const
    {
        std::vector<long> v;
        for (const Level& L : levels_)
        {
            v.push_back(static_cast<long>(L.alpha->boxArray().numPts()));
        }
        return v;
    }

private:

    // Templated on the fab value type so the field fabs and the (possibly narrower)
    // coefficient fabs share one allocation path.
    template<class V>
    static std::unique_ptr<la::GmgFab<V>>
    makeFab(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int ng)
    {
        auto mf = std::make_unique<la::GmgFab<V>>(ba, dm, 1, ng);
        mf->setVal(V(0));
        return mf;
    }

    static std::unique_ptr<Fab>
    makeMf(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int ng)
    {
        return makeFab<T>(ba, dm, ng);
    }

    static std::unique_ptr<CoeffFab>
    makeCoeffMf(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int ng)
    {
        return makeFab<TC>(ba, dm, ng);
    }

    static Level makeLevel(
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        const amrex::Geometry& geom,
        bool shared
    )
    {
        Level L;
        L.geom = geom;
        L.alpha = makeCoeffMf(ba, dm, 0);
        const auto fba = [&ba](int d)
        { return amrex::convert(ba, amrex::IntVect::TheDimensionVector(d)); };
        L.ux = makeCoeffMf(fba(0), dm, 0);
        L.uy = makeCoeffMf(fba(1), dm, 0);
        L.uz = makeCoeffMf(fba(2), dm, 0);
        if (!shared)
        {
            L.lx = makeCoeffMf(fba(0), dm, 0);
            L.ly = makeCoeffMf(fba(1), dm, 0);
            L.lz = makeCoeffMf(fba(2), dm, 0);
        }
        L.sol = makeMf(ba, dm, 1);
        L.rhs = makeMf(ba, dm, 0);
        return L;
    }

    // The caller's fields converted into level 0's OWN fabs, read at setup only: later
    // in-place writes by the caller are not seen, which is fine because a changed
    // operator means a rebuilt preconditioner.
    static void copyCallerCoeffs(const GmgArgs& args, Level& L)
    {
        la::gmgConvertCopy(*L.alpha, *args.alpha, /*onDevice=*/true);
        la::gmgConvertCopy(*L.ux, *args.ux, /*onDevice=*/true);
        la::gmgConvertCopy(*L.uy, *args.uy, /*onDevice=*/true);
        la::gmgConvertCopy(*L.uz, *args.uz, /*onDevice=*/true);
        if (!L.shared())
        {
            la::gmgConvertCopy(*L.lx, *args.lx, /*onDevice=*/true);
            la::gmgConvertCopy(*L.ly, *args.ly, /*onDevice=*/true);
            la::gmgConvertCopy(*L.lz, *args.lz, /*onDevice=*/true);
        }
    }

    // Rediscretise the operator on the coarse level: faces through gmgCoarsenFace (the
    // 4 fine faces averaged, then rescaled for the doubled dx), alpha through
    // gmgRestrict's plain 8-child average -- correct only because alpha is a
    // dx-independent density. Both fabs must share a DistributionMapping and box order.
    // A shared level does three faces instead of six: averaging the same fine values
    // twice gives the same coarse numbers, so symmetry survives the hierarchy and the
    // pair is never re-formed.
    static void restrictCoeffs(const Level& f, Level& c)
    {
        la::gmgRestrict<TC>(*f.alpha, *c.alpha, /*onDevice=*/true);
        la::gmgCoarsenFace<TC>(*f.ux, *c.ux, 0, 4.0, /*onDevice=*/true);
        la::gmgCoarsenFace<TC>(*f.uy, *c.uy, 1, 4.0, /*onDevice=*/true);
        la::gmgCoarsenFace<TC>(*f.uz, *c.uz, 2, 4.0, /*onDevice=*/true);
        if (!c.shared())
        {
            la::gmgCoarsenFace<TC>(f.lxf(), *c.lx, 0, 4.0, /*onDevice=*/true);
            la::gmgCoarsenFace<TC>(f.lyf(), *c.ly, 1, 4.0, /*onDevice=*/true);
            la::gmgCoarsenFace<TC>(f.lzf(), *c.lz, 2, 4.0, /*onDevice=*/true);
        }
    }

    // Move the rediscretised coefficients onto another decomposition of the same
    // region. Face BoxArrays overlap on internal faces, but a shared face carries one
    // value, so which source box wins does not matter.
    static void copyCoeffs(const Level& src, Level& dst)
    {
        dst.alpha->ParallelCopy(*src.alpha, 0, 0, 1);
        dst.ux->ParallelCopy(*src.ux, 0, 0, 1);
        dst.uy->ParallelCopy(*src.uy, 0, 0, 1);
        dst.uz->ParallelCopy(*src.uz, 0, 0, 1);
        if (!dst.shared())
        {
            dst.lx->ParallelCopy(src.lxf(), 0, 0, 1);
            dst.ly->ParallelCopy(src.lyf(), 0, 0, 1);
            dst.lz->ParallelCopy(src.lzf(), 0, 0, 1);
        }
    }

    // Periodic/internal ghosts, then the homogeneous physical-BC reflection — the same
    // two steps in the same order as production's fillGhosts. On a periodic mesh the
    // second step has no tasks.
    void fillGhosts(Level& L) const
    {
        if (amrexFree_)
        {
            gmgFillBoundaryKokkos<T>(*L.sol, L.halo);
            if (hasPhysBc_)
            {
                gmgFillDomainBcKokkos<T>(*L.sol, L.bc);
            }
            return;
        }
        // AMReX is about to READ sol, which the last colour sweep wrote: a no-op for
        // the fenced backends, the dropped fence's ordering for kokkos_opt on >1 rank.
        Backend::beforeAmrexRead();
        L.sol->FillBoundary(L.geom.periodicity());
        if (hasPhysBc_)
        {
            la::fillDomainBcGhostsDevice(*L.sol, L.geom.Domain(), bc_);
        }
        Backend::afterAmrexWrite();
    }

    // Red-black colour sweeps; `reversed` flips to black-red, the adjoint of the
    // forward sweep, so the whole V-cycle stays symmetric.
    void smooth(std::size_t l, int sweeps, bool reversed)
    {
        Level& L = levels_[l];
        for (int s = 0; s < sweeps; ++s)
        {
            for (int c = 0; c < 2; ++c)
            {
                const int parity = (reversed ? 1 + c : c) & 1;
                fillGhosts(L); // the other colour changed — refresh ghosts
                Backend::gsColor(
                    *L.sol,
                    *L.rhs,
                    *L.ux,
                    L.lxf(),
                    *L.uy,
                    L.lyf(),
                    *L.uz,
                    L.lzf(),
                    *L.alpha,
                    parity,
                    omega_
                );
            }
        }
    }

    void vcycle(std::size_t l)
    {
        Level& L = levels_[l];
        if (l + 1 == levels_.size())
        {
            // Forward + reversed halves keep the coarsest "solve" self-adjoint.
            smooth(l, coarsestSweeps_ / 2, false);
            smooth(l, coarsestSweeps_ / 2, true);
            return;
        }
        smooth(l, preSweeps_, false);
        fillGhosts(L);
        Level& C = levels_[l + 1];
        // On an agglomerated level the kernels write/read the transfer fab on this
        // level's coarsened layout, and a copy bridges to the coarse decomposition.
        Backend::residRestrict(
            *L.sol,
            *L.rhs,
            C.agglomerated ? *C.xferRhs : *C.rhs,
            *L.ux,
            L.lxf(),
            *L.uy,
            L.lyf(),
            *L.uz,
            L.lzf(),
            *L.alpha
        );
        if (amrexFree_)
        {
            if (C.agglomerated)
            {
                gmgCopyKokkos<T>(*C.rhs, *C.xferRhs, C.xferIn);
            }
            gmgZeroKokkos<T>(*C.sol);
        }
        else
        {
            Backend::beforeAmrexRead(); // residRestrict just wrote xferRhs
            if (C.agglomerated)
            {
                C.rhs->ParallelCopy(*C.xferRhs, 0, 0, 1);
            }
            C.sol->setVal(T(0));
            Backend::afterAmrexWrite();
        }
        vcycle(l + 1);
        if (C.agglomerated)
        {
            if (amrexFree_)
            {
                gmgCopyKokkos<T>(*C.xferSol, *C.sol, C.xferOut);
            }
            else
            {
                Backend::beforeAmrexRead(); // the coarse cycle just wrote C.sol
                C.xferSol->ParallelCopy(*C.sol, 0, 0, 1);
                Backend::afterAmrexWrite();
            }
        }
        Backend::prolongAdd(C.agglomerated ? *C.xferSol : *C.sol, *L.sol);
        smooth(l, postSweeps_, true);
    }

    int preSweeps_;
    int postSweeps_;
    int coarsestSweeps_;
    double omega_;
    la::BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool shared_ = false;

    // Whether the timed cycle really is AMReX-free: the backend must offer the Kokkos
    // data movements AND every box they name must be local, which on >1 rank it is not
    // (see KokkosOptGmgBackend). False changes no arithmetic -- it routes the halo, the
    // zero fill and the agglomeration transfers through AMReX instead.
    bool amrexFree_ = false;

    // Level-0 agglomeration only: the caller-layout staging fab and its plans. Empty
    // otherwise, and the apply path then talks to level 0 directly.
    bool aggL0_ = false;
    std::unique_ptr<Fab> iface_;
    CopyPlan ifaceIn_, ifaceOut_;
    std::vector<Level> levels_;
};

// The six (field, coefficient) pairs. parseCoeffPrecision already rejected a
// coefficient wider than the field, so the other three cells of the 3x3 never arrive
// here.
enum class PrecPair
{
    f64c64,
    f64c32,
    f64c16,
    f32c32,
    f32c16,
    f16c16
};

PrecPair precPair(Precision field, Precision coeff)
{
    switch (field)
    {
    case Precision::fp64:
        switch (coeff)
        {
        case Precision::fp32:
            return PrecPair::f64c32;
        case Precision::bf16:
            return PrecPair::f64c16;
        case Precision::fp64:
            break;
        }
        return PrecPair::f64c64;
    case Precision::fp32:
        return coeff == Precision::bf16 ? PrecPair::f32c16 : PrecPair::f32c32;
    case Precision::bf16:
        break;
    }
    return PrecPair::f16c16;
}

} // namespace

} // namespace blockamr
