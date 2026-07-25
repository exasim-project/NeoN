// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The GMG V-cycle bench: the native geometric-multigrid V-cycle of
// solvers/gmg_precond.hpp, run with its AMReX kernels and with the Kokkos twins in
// gmg_kokkos.hpp. Same hierarchy, same sweep counts, same control flow, same order
// of operations — only the launcher differs, which is the same discipline the
// operator bench uses.
//
// Four backends, each the previous one plus one change, so a row of the bench is
// read against the row above it:
//
//   amrex         the production per-box path — the orientation point.
//   kokkos        its per-box Kokkos twin.
//   kokkos_fused  the same kernels under one TeamPolicy launch per level.
//   kokkos_opt    ... and the halo exchange, the zero fill and the agglomeration
//                 transfers on Kokkos too (halo_kokkos.hpp), which leaves no AMReX
//                 operation inside the timed cycle and therefore no reason to fence
//                 between kernels at all. The whole cycle becomes one stream the
//                 host can run ahead of.
//
// Only the Kokkos side is optimised, deliberately — the AMReX column has to stay the
// shipped V-cycle for the comparison to mean anything, and `kokkos`/`kokkos_fused`
// stay put as the intermediate baselines. Orthogonal to the backend,
// GmgArgs::agglomerate switches the hierarchy from production's in-place BoxArray
// coarsening to a re-decomposed coarse grid; since red-black smoothing is
// decomposition-independent, at equal depth that changes cost without changing a
// single arithmetic result.
//
// This is a port of the DEVICE path of GmgPrecondT<double>, reduced to what the
// timed V-cycle needs and nothing more:
//
//   kept     hierarchy construction by in-place BoxArray coarsening (so box COUNT
//            is preserved down the levels, exactly as in production, unless
//            agglomeration is asked for), RB-SOR smoothing with the reversed
//            post-sweep, fused residual+restriction,
//            piecewise-constant prolongation, the ghost fill per colour, and the
//            recursive V-cycle with warm-started sol.
//   dropped  Ginkgo (no LinOp, no Dense pack/unpack), the ReferenceExecutor host
//            path, the Chebyshev smoother and
//            its λmax power iteration, and physical boundary conditions — the
//            bench is triply periodic, so bc handling never fires and solvers/bc
//            stays out of this translation unit.
//
// The AMReX column calls the PRODUCTION kernels (solvers/gmg_kernels.hpp) rather
// than a copy of them, so the baseline is the real thing. It is recompiled here in
// the non-RDC object library, which is what makes the flags identical for both
// columns (see CMakeLists.txt); it does mean the AMReX kernels here are non-RDC
// while production ones are RDC.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include <AMReX_MultiFab.H>
#include <AMReX_Reduce.H>

#include "../solvers/bc_geom.hpp"
#include "../solvers/gmg_kernels.hpp"
#include "../solvers/transfer.hpp"
#include "gmg_apply.hpp"
#include "gmg_kokkos.hpp"
#include "halo_kokkos.hpp"
#include "kokkos_bench.hpp"

namespace blockamr::bench
{

namespace
{

// The level scalar is a template parameter, as in production's GmgPrecondT: the whole
// hierarchy below level 0 can be carried in fp32 while the operator and the caller's
// fields stay fp64, halving the traffic of a smoother that is bandwidth-bound at
// scale. Only kokkos_opt is instantiated for float — the baselines stay fp64, which
// is what keeps them baselines.

// ---------------------------------------------------------------------------
// The backends. Each is the three kernels the timed V-cycle runs plus the two
// cross-runtime ordering points, and nothing else, so a backend cannot quietly
// differ in anything but what it is meant to.
//
//   afterAmrexWrite   order an AMReX write against a following backend kernel.
//   beforeAmrexRead   order a backend kernel against a following AMReX read.
//   amrexFreeCycle    the timed cycle contains no AMReX operation, so the kernels
//                     need no fence between them (they share one stream) and the
//                     data movements come from halo_kokkos.hpp.
// ---------------------------------------------------------------------------

struct AmrexGmgBackend
{
    static constexpr const char* tag = "amrex";
    static constexpr bool canShareCoeffs = false;
    static constexpr bool amrexFreeCycle = false;

    // AMReX kernels are issued to AMReX's own stream, so an AMReX write before them
    // (FillBoundary, setVal) is already ordered against them: nothing to do. Nor is
    // there anything to do the other way round.
    static void afterAmrexWrite() {}

    static void beforeAmrexRead() {}

    template<class... A>
    static void gsColor(A&&... a)
    {
        solvers::gmgGsColorDevice<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void residRestrict(A&&... a)
    {
        solvers::gmgResidRestrictDevice<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void prolongAdd(A&&... a)
    {
        solvers::gmgProlongAddDevice<double>(std::forward<A>(a)...);
    }
};

struct KokkosGmgBackend
{
    static constexpr const char* tag = "kokkos";
    static constexpr bool canShareCoeffs = false;
    static constexpr bool amrexFreeCycle = false;

    // Every kernel already fences, so a following AMReX read is ordered.
    static void beforeAmrexRead() {}

    // The other half of straddling two runtimes (the Kokkos::fence at the end of
    // each ported kernel is the first half): a Kokkos kernel about to read what
    // AMReX just wrote has no ordering against AMReX's stream, so the write has to
    // be waited on. This is the one thing the port cannot express in Kokkos, and it
    // doubles the host syncs per colour -- one for FillBoundary, one for the kernel
    // -- where the AMReX path needs only the one MFIter already performs.
    static void afterAmrexWrite() { amrex::Gpu::streamSynchronizeAll(); }

    template<class... A>
    static void gsColor(A&&... a)
    {
        gmgGsColorKokkos<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void residRestrict(A&&... a)
    {
        gmgResidRestrictKokkos<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void prolongAdd(A&&... a)
    {
        gmgProlongAddKokkos<double>(std::forward<A>(a)...);
    }
};

// The same three kernels under ONE launch per level instead of one per box. Only
// the Kokkos side gets a fused variant: the AMReX column stays the production
// per-box path, so it remains the orientation point every other column is read
// against.
struct KokkosFusedGmgBackend
{
    static constexpr const char* tag = "kokkos_fused";
    static constexpr bool canShareCoeffs = false;
    static constexpr bool amrexFreeCycle = false;

    static void beforeAmrexRead() {}

    static void afterAmrexWrite() { amrex::Gpu::streamSynchronizeAll(); }

    template<class... A>
    static void gsColor(A&&... a)
    {
        gmgGsColorKokkosFused<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void residRestrict(A&&... a)
    {
        gmgResidRestrictKokkosFused<double>(std::forward<A>(a)...);
    }

    template<class... A>
    static void prolongAdd(A&&... a)
    {
        gmgProlongAddKokkosFused<double>(std::forward<A>(a)...);
    }
};

// The fused kernels again, with every AMReX operation removed from the timed cycle
// (halo_kokkos.hpp supplies the halo exchange, the zero fill and the agglomeration
// transfers) and therefore with the per-kernel fence removed too: consecutive Kokkos
// kernels on one execution space are already ordered by the stream. What that buys is
// not the fences themselves but the serialisation they enforced -- the host no longer
// waits on the device inside a cycle, so it can run the launch of a coarse level
// ahead of the arithmetic of the level above it.
//
// The fence could not be dropped before this: each colour sweep was followed by an
// AMReX FillBoundary, which needs exactly the same ordering the fence provided. The
// halo port is what makes the removal possible, not an optimisation beside it.
struct KokkosOptGmgBackend
{
    static constexpr const char* tag = "kokkos_opt";
    static constexpr bool amrexFreeCycle = true;
    static constexpr bool canShareCoeffs = true;

    // The reset/residual path still crosses into AMReX, outside the timed region.
    static void beforeAmrexRead() { Kokkos::fence(); }

    static void afterAmrexWrite() { amrex::Gpu::streamSynchronizeAll(); }

    // The level scalar is deduced from the fabs rather than fixed at double, which is
    // what lets this backend — and only this one — be instantiated for fp32.
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

// Are these two fields the same numbers everywhere?
//
// The question this answers is whether the operator is symmetric. Cell i's east
// coefficient is stored at face i+1 of ux; cell i+1's west coefficient is stored at
// face i+1 of lx; and they are the two off-diagonal entries A[i][i+1] and A[i+1][i].
// So ux == lx pointwise IS symmetry, and it is the exact condition under which one
// array can stand in for both.
//
// Bitwise, deliberately: near-equal is a different operator, and a tolerance here
// would silently symmetrise it. The common case is the same fab passed twice (the
// solver hands ux=lx=fx), which the pointer test settles without a kernel; setup
// only, so the reduction costs nothing that gets timed.
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
    const double diff = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpMax> {},
        amrex::TypeList<double> {},
        a,
        amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) -> amrex::GpuTuple<double>
        { return {amrex::Math::abs(aa[box](i, j, k) - bb[box](i, j, k))}; }
    );
    return diff == 0.0;
}

// One multigrid level, as in GmgLevelT: geometry, rediscretised coefficients and
// preallocated work fields (sol needs 1 ghost for the stencil, rhs is valid-only).
template<class T>
struct LevelT
{
    // The level fab type production uses (FabArray<BaseFab<T>>, not MultiFab), so the
    // production kernels apply unchanged.
    using Fab = solvers::GmgFab<T>;

    amrex::Geometry geom;
    std::unique_ptr<Fab> alpha, ux, lx, uy, ly, uz, lz, sol, rhs;

    // lx/ly/lz are NULL on a level that shares one face coefficient per direction
    // (GmgArgs::shareCoeffs, kokkos_opt only). The stencil reads the east coefficient
    // at face i+1 and the west at face i, so for a symmetric operator -- where the
    // two fabs hold identical numbers -- handing the kernels ux for both arguments is
    // the same arithmetic on half the storage. Read the lower coefficients through
    // these accessors, never through the pointers, so a shared level cannot be
    // dereferenced by accident.
    [[nodiscard]] const Fab& lxf() const { return lx ? *lx : *ux; }
    [[nodiscard]] const Fab& lyf() const { return ly ? *ly : *uy; }
    [[nodiscard]] const Fab& lzf() const { return lz ? *lz : *uz; }
    [[nodiscard]] bool shared() const { return lx == nullptr; }

    // Agglomerated levels only. This level's BoxArray is a fresh decomposition of
    // its domain rather than the fine level's coarsened in place, so it no longer
    // matches the fine level box for box and the inter-level kernels -- which
    // address a fine and a coarse fab at the SAME local box index -- cannot reach it
    // directly. These two hold the restriction's output and the prolongation's input
    // on coarsen(fine BoxArray, 2) with the FINE DistributionMapping, which is the
    // layout those kernels can address; AMReX ParallelCopy moves between them and
    // this level's own fields.
    std::unique_ptr<Fab> xferRhs, xferSol;
    bool agglomerated = false;

    // kokkos_opt only, and empty for every other backend: the data movements of this
    // level resolved to device tables at setup. `halo` is the ghost exchange of sol,
    // `bc` the homogeneous domain-boundary reflection (empty on a periodic mesh, and
    // on any level whose boxes touch no physical face), `xferIn`/`xferOut` the two
    // directions of the agglomeration transfer.
    CopyPlan halo, bc, xferIn, xferOut;
};

template<class Backend, class T>
class Vcycle
{
public:

    using Level = LevelT<T>;
    using Fab = solvers::GmgFab<T>;

    Vcycle(const GmgArgs& args)
        : preSweeps_(args.preSweeps), postSweeps_(args.postSweeps),
          coarsestSweeps_(args.coarsestSweeps), omega_(args.omega), bc_(args.bc),
          hasPhysBc_(std::any_of(args.bc.begin(), args.bc.end(), [](int b) { return b != 0; }))
    {
        const amrex::BoxArray& ba = args.alpha->boxArray();
        const amrex::DistributionMapping& dm = args.alpha->DistributionMap();

        // One face coefficient per direction instead of an upper/lower pair, when the
        // caller asked for it, the backend supports it AND the operator really is
        // symmetric. Checked rather than assumed: ux == lx is what makes the two fabs
        // interchangeable, and an asymmetric operator that quietly lost its lower
        // coefficients would solve a different system at full speed.
        shared_ = Backend::canShareCoeffs && args.shareCoeffs && sameField(*args.ux, *args.lx)
               && sameField(*args.uy, *args.ly) && sameField(*args.uz, *args.lz);

        levels_.push_back(makeLevel(ba, dm, *args.geom, shared_));
        solvers::gmgConvertCopyDevice(*levels_[0].alpha, *args.alpha);
        solvers::gmgConvertCopyDevice(*levels_[0].ux, *args.ux);
        solvers::gmgConvertCopyDevice(*levels_[0].uy, *args.uy);
        solvers::gmgConvertCopyDevice(*levels_[0].uz, *args.uz);
        if (!shared_)
        {
            solvers::gmgConvertCopyDevice(*levels_[0].lx, *args.lx);
            solvers::gmgConvertCopyDevice(*levels_[0].ly, *args.ly);
            solvers::gmgConvertCopyDevice(*levels_[0].lz, *args.lz);
        }

        // Coarsen while the BoxArray stays coarsenable and the coarse domain keeps
        // >= minBottom cells per direction. Without agglomeration the fine BoxArray
        // is coarsened in place and the DistributionMapping reused, so the number of
        // boxes is the same on every level and only their size shrinks — the
        // production behaviour, and the reason the coarsest level is launch-bound.
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
            const amrex::Geometry cgeom(
                cdom,
                fgeom.ProbDomain(),
                fgeom.Coord(),
                {fgeom.isPeriodic(0), fgeom.isPeriodic(1), fgeom.isPeriodic(2)}
            );

            // Agglomeration: take a fresh aggGridSize-capped decomposition of the
            // coarse domain when it has strictly fewer boxes than the in-place
            // coarsening. Same mechanism MLMG uses (LPInfo::do_agglomeration
            // rebuilds the coarse grids as BoxArray(domain).maxSize(agg_grid_size),
            // AMReX_MLLinOp.H:1028) but not the same trigger: MLMG agglomerates once
            // the average box falls below agg_grid_size^3 cells, with a default
            // agg_grid_size of 8 in 3D, because what it is reducing is the number of
            // MPI ranks with work. On one GPU that default would leave the box count
            // untouched; the cost here is per-box kernel launches, so the trigger is
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
                // rediscretise there and copy the result onto this level's
                // decomposition. Setup, so the extra fabs are transient and untimed.
                Level t = makeLevel(cba, fdm, cgeom, shared_);
                restrictCoeffs(fl, t);
                copyCoeffs(t, c);
            }
        }

        // Resolve the data movements once, now that the hierarchy is final. Setup, so
        // untimed: what the timed cycle sees is a device table and one launch.
        if constexpr (Backend::amrexFreeCycle)
        {
            for (Level& L : levels_)
            {
                L.halo = makeHaloPlan(*L.sol, L.geom.periodicity());
                if (hasPhysBc_)
                {
                    // The coarse domains are the fine one coarsened, so every level
                    // has the same physical faces and the same bc spec applies
                    // throughout -- as in production (gmg_precond.hpp fillGhosts).
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
        solvers::gmgConvertCopyDevice(*levels_[0].rhs, rhs);
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
    // V-cycles, z <- L0 sol. This is the same sequence GmgPrecondT::apply_impl runs
    // (gmg_precond.hpp:304), including the two AMReX transfers -- so it pays the same
    // cross-runtime sync at each end, and a solver-level comparison measures the
    // cycle rather than a difference in plumbing. Inside, the cycle stays fence-free.
    void applyFlat(const double* r, double* z, int nCycles)
    {
        Level& L0 = levels_.front();
        solvers::scatter_device(r, *L0.rhs);
        L0.sol->setVal(T(0)); // z0 = 0: apply M^{-1}, not a warm-started solve
        Backend::afterAmrexWrite();
        cycles(nCycles);
        Backend::beforeAmrexRead();
        solvers::gather_device(*L0.sol, z, 1.0);
        amrex::Gpu::streamSynchronize(); // z complete before the caller reads it
    }

    // sum((rhs - A sol)^2) on the finest level. Reporting only: it is the gate that
    // says the timed V-cycle did the work, so it stays an AMReX reduction for both
    // backends and never runs inside a timed region.
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
        return amrex::ParReduce(
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
    }

    int nlevels() const { return static_cast<int>(levels_.size()); }

    // What the hierarchy actually does, not what was requested (see sameField).
    bool sharedCoeffs() const { return shared_; }

    // Boxes and cells PER LEVEL: the point of the exercise is that the box count
    // does not shrink while the cell count does.
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

    static std::unique_ptr<Fab>
    makeMf(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int ng)
    {
        auto mf = std::make_unique<Fab>(ba, dm, 1, ng);
        mf->setVal(T(0));
        return mf;
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
        L.alpha = makeMf(ba, dm, 0);
        const auto fba = [&ba](int d)
        { return amrex::convert(ba, amrex::IntVect::TheDimensionVector(d)); };
        L.ux = makeMf(fba(0), dm, 0);
        L.uy = makeMf(fba(1), dm, 0);
        L.uz = makeMf(fba(2), dm, 0);
        if (!shared)
        {
            L.lx = makeMf(fba(0), dm, 0);
            L.ly = makeMf(fba(1), dm, 0);
            L.lz = makeMf(fba(2), dm, 0);
        }
        L.sol = makeMf(ba, dm, 1);
        L.rhs = makeMf(ba, dm, 0);
        return L;
    }

    // Rediscretise the operator on the coarse level: volume-average the diagonal
    // source, area-average the face coefficients (4 fine faces per coarse face).
    // Both fabs must share a DistributionMapping and box order.
    // A shared level rediscretises three faces instead of six -- area-averaging the
    // same fine values twice would produce the same coarse numbers, so symmetry is
    // preserved down the hierarchy and the pair never has to be re-formed.
    static void restrictCoeffs(const Level& f, Level& c)
    {
        solvers::gmgRestrictDevice<T>(*f.alpha, *c.alpha);
        solvers::gmgCoarsenFaceDevice<T>(*f.ux, *c.ux, 0, 4.0);
        solvers::gmgCoarsenFaceDevice<T>(*f.uy, *c.uy, 1, 4.0);
        solvers::gmgCoarsenFaceDevice<T>(*f.uz, *c.uz, 2, 4.0);
        if (!c.shared())
        {
            solvers::gmgCoarsenFaceDevice<T>(f.lxf(), *c.lx, 0, 4.0);
            solvers::gmgCoarsenFaceDevice<T>(f.lyf(), *c.ly, 1, 4.0);
            solvers::gmgCoarsenFaceDevice<T>(f.lzf(), *c.lz, 2, 4.0);
        }
    }

    // Move the rediscretised coefficients onto a different decomposition of the same
    // region. The face BoxArrays overlap on internal faces, but a shared face carries
    // one value, so which source box wins does not matter.
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

    // Periodic/internal ghosts, then the homogeneous physical-BC reflection — the
    // same two steps in the same order as the production fillGhosts. On the bench's
    // own triply periodic mesh the second step has no tasks and is skipped entirely,
    // so the measured backends are unaffected by its existence.
    void fillGhosts(Level& L) const
    {
        if constexpr (Backend::amrexFreeCycle)
        {
            gmgFillBoundaryKokkos<T>(*L.sol, L.halo);
            if (hasPhysBc_)
            {
                gmgFillDomainBcKokkos<T>(*L.sol, L.bc);
            }
        }
        else
        {
            L.sol->FillBoundary(L.geom.periodicity());
            if (hasPhysBc_)
            {
                solvers::fillDomainBcGhostsDevice(*L.sol, L.geom.Domain(), bc_);
            }
            Backend::afterAmrexWrite();
        }
    }

    // Red-black colour sweeps; `reversed` flips the colour order (black-red), the
    // adjoint of the forward sweep, so the whole V-cycle stays symmetric.
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
            // Tiny grid: forward + reversed halves keep the coarsest "solve"
            // self-adjoint.
            smooth(l, coarsestSweeps_ / 2, false);
            smooth(l, coarsestSweeps_ / 2, true);
            return;
        }
        smooth(l, preSweeps_, false);
        fillGhosts(L);
        Level& C = levels_[l + 1];
        // On an agglomerated level the kernels write/read the transfer fab, which
        // lives on this level's coarsened layout, and a copy bridges to and from the
        // coarse decomposition — AMReX ParallelCopy, or its Kokkos twin under
        // kokkos_opt.
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
        if constexpr (Backend::amrexFreeCycle)
        {
            if (C.agglomerated)
            {
                gmgCopyKokkos<T>(*C.rhs, *C.xferRhs, C.xferIn);
            }
            gmgZeroKokkos<T>(*C.sol);
        }
        else
        {
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
            if constexpr (Backend::amrexFreeCycle)
            {
                gmgCopyKokkos<T>(*C.xferSol, *C.sol, C.xferOut);
            }
            else
            {
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
    solvers::BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool shared_ = false;
    std::vector<Level> levels_;
};

// Fence both runtimes regardless of backend, as in benchOperator.
void fenceAll()
{
    amrex::Gpu::streamSynchronize();
    if (Kokkos::is_initialized())
    {
        Kokkos::fence();
    }
}

template<class Backend, class T>
GmgResult run(const GmgArgs& args, int iters, int batches)
{
    Vcycle<Backend, T> v(args);

    GmgResult r;
    r.nlevels = v.nlevels();
    r.sharedCoeffs = v.sharedCoeffs();
    r.boxesPerLevel = v.boxesPerLevel();
    r.cellsPerLevel = v.cellsPerLevel();

    // Correctness/strength gate, untimed: how far ONE V-cycle from z0 = 0 moves the
    // residual. A launcher that indexes wrongly cannot reproduce this number.
    v.reset(*args.rhs);
    r.resid0 = std::sqrt(v.residSumSq());
    v.cycles(1);
    r.resid1 = std::sqrt(v.residSumSq());

    // Timed: every batch starts from the same state, so a batch measures
    // `iters` V-cycles of the same work rather than an ever-converging one.
    v.reset(*args.rhs);
    v.cycles(1);
    fenceAll();

    std::vector<double> ms, msEnq;
    ms.reserve(static_cast<std::size_t>(batches));
    msEnq.reserve(static_cast<std::size_t>(batches));
    for (int b = 0; b < batches; ++b)
    {
        v.reset(*args.rhs);
        fenceAll();
        const auto t0 = std::chrono::steady_clock::now();
        v.cycles(iters);
        const auto t1 = std::chrono::steady_clock::now();
        fenceAll();
        const auto t2 = std::chrono::steady_clock::now();
        msEnq.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
        ms.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count() / iters);
    }
    std::sort(ms.begin(), ms.end());
    std::sort(msEnq.begin(), msEnq.end());
    r.msMin = ms.front();
    r.msMedian = ms[ms.size() / 2];
    r.msEnqueue = msEnq.front();
    return r;
}

// The optimised V-cycle behind the Ginkgo-free handle of gmg_apply.hpp. Fixed to
// KokkosOptGmgBackend: a caller wanting the baselines has the bench for that, and a
// preconditioner has no reason to run a deliberately unoptimised launcher.
template<class T>
class KokkosGmgApplyImpl final : public KokkosGmgApply
{
public:

    KokkosGmgApplyImpl(const GmgArgs& args, int nCycles) : v_(args), nCycles_(nCycles) {}

    void apply(const double* r, double* z) override { v_.applyFlat(r, z, nCycles_); }

    [[nodiscard]] int nlevels() const override { return v_.nlevels(); }

private:

    Vcycle<KokkosOptGmgBackend, T> v_;
    int nCycles_;
};

} // namespace

std::unique_ptr<KokkosGmgApply> makeKokkosGmgApply(
    const amrex::Geometry& geom,
    const amrex::MultiFab& alpha,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const KokkosGmgOpts& opts
)
{
    // No bc/geometry consistency check here: solvers::parseBc already refuses a
    // non-periodic direction marked periodic and a periodic one marked otherwise, and
    // it is the only path that reaches this factory. Repeating it would be a branch no
    // test could reach.
    GmgArgs args;
    args.geom = &geom;
    args.rhs = nullptr; // the hierarchy is built from the coefficients alone
    args.alpha = &alpha;
    args.ux = &ux;
    args.lx = &lx;
    args.uy = &uy;
    args.ly = &ly;
    args.uz = &uz;
    args.lz = &lz;
    args.preSweeps = opts.preSweeps;
    args.postSweeps = opts.postSweeps;
    args.coarsestSweeps = opts.coarsestSweeps;
    args.maxLevels = opts.maxLevels;
    args.minBottom = opts.minBottom;
    args.omega = opts.omega;
    args.agglomerate = opts.agglomerate;
    args.aggGridSize = opts.aggGridSize;
    args.fp32 = opts.fp32;
    args.shareCoeffs = opts.shareCoeffs;
    args.bc = opts.bc;

    if (opts.fp32)
    {
        return std::make_unique<KokkosGmgApplyImpl<float>>(args, opts.cycles);
    }
    return std::make_unique<KokkosGmgApplyImpl<double>>(args, opts.cycles);
}

std::vector<std::string> benchGmgBackends()
{
    return {
        AmrexGmgBackend::tag,
        KokkosGmgBackend::tag,
        KokkosFusedGmgBackend::tag,
        KokkosOptGmgBackend::tag
    };
}

GmgResult benchGmgVcycle(const std::string& backend, const GmgArgs& args, int iters, int batches)
{
    // Before the dispatch, not after: silently ignoring fp32 on a backend that has no
    // fp32 hierarchy would report an fp64 timing under an fp32 label.
    if (args.fp32 && backend != KokkosOptGmgBackend::tag)
    {
        throw std::runtime_error(
            "benchGmgVcycle: fp32 is implemented for the '" + std::string(KokkosOptGmgBackend::tag)
            + "' backend only, not '" + backend + "'"
        );
    }
    // Same reason: a baseline silently ignoring share_coeffs would report the
    // unshared timing under a shared label.
    if (args.shareCoeffs && backend != KokkosOptGmgBackend::tag)
    {
        throw std::runtime_error(
            "benchGmgVcycle: share_coeffs is implemented for the '"
            + std::string(KokkosOptGmgBackend::tag) + "' backend only, not '" + backend + "'"
        );
    }
    if (backend == AmrexGmgBackend::tag)
    {
        return run<AmrexGmgBackend, double>(args, iters, batches);
    }
    if (backend == KokkosGmgBackend::tag)
    {
        return run<KokkosGmgBackend, double>(args, iters, batches);
    }
    if (backend == KokkosFusedGmgBackend::tag)
    {
        return run<KokkosFusedGmgBackend, double>(args, iters, batches);
    }
    if (backend == KokkosOptGmgBackend::tag)
    {
        return args.fp32 ? run<KokkosOptGmgBackend, float>(args, iters, batches)
                         : run<KokkosOptGmgBackend, double>(args, iters, batches);
    }
    if (args.fp32)
    {
        throw std::runtime_error(
            "benchGmgVcycle: fp32 is implemented for the '" + std::string(KokkosOptGmgBackend::tag)
            + "' backend only, not '" + backend + "'"
        );
    }
    throw std::runtime_error("benchGmgVcycle: unknown backend '" + backend + "'");
}

} // namespace blockamr::bench
