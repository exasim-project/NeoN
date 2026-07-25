// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ---------------------------------------------------------------------------
// The optimised (kokkos_opt) V-cycle as a preconditioner apply, behind an interface
// that mentions neither Kokkos nor Ginkgo.
//
// That is the point of this header, and it is a build constraint rather than a taste:
// the Kokkos kernels compile in the non-RDC blockamr_bench object library (AMReX puts
// _blockamr in -rdc=true mode and Kokkos' desul atomics refuse it), while the Ginkgo
// solver stack compiles in the RDC one. Neither library can include the other's
// headers. So the V-cycle is exposed here as an opaque handle over flat device
// vectors, and solvers/gmg_kokkos_precond.hpp wraps it in a gko::LinOp on the other
// side of the fence -- Ginkgo there, Kokkos here, nothing shared but this header and
// two double pointers.
//
// Why bother: bench_gmg_kokkos.py measures the V-cycle in isolation, but a V-cycle is
// a preconditioner. What a caller cares about is the SOLVE, where the V-cycle is one
// term next to the matrix-vector product, the Krylov vector algebra and the iteration
// count. Handing the optimised cycle to the real CG is the only way to find out what
// its 3.17x is worth end to end, and the only way to compare it with MLMG on equal
// terms -- bench_solvers.py already runs MLMG, matrix-free CG, MLMG-preconditioned CG,
// the native GMG preconditioner and its Ir twin over the same operator.
// ---------------------------------------------------------------------------

namespace blockamr::bench
{

// The V-cycle shape, matching GmgArgs (kokkos_bench.hpp) and production's
// gmg_* solver options.
struct KokkosGmgOpts
{
    int cycles = 1;
    int preSweeps = 2;
    int postSweeps = 2;
    int coarsestSweeps = 8;
    int maxLevels = 0; // 0 = coarsen as far as the grid allows
    int minBottom = 2;
    double omega = 1.0;
    bool fp32 = false;

    // On by default here, unlike in the bench. It cannot change the result at equal
    // depth (red-black smoothing is decomposition-independent) and it is what keeps
    // the coarse levels from being one launch per tiny box; the depth it additionally
    // unlocks is a better-coarsened hierarchy, which the iteration count will show.
    bool agglomerate = true;
    int aggGridSize = 32;
};

// z = M^{-1} r, on flat DEVICE vectors in the solver's cell ordering (MFIter order,
// i fastest within a valid box -- the ordering solvers/transfer.hpp defines and the
// whole Ginkgo stack already uses). Always fp64 on the outside, whatever the
// hierarchy carries inside.
class KokkosGmgApply
{
public:

    KokkosGmgApply() = default;
    virtual ~KokkosGmgApply() = default;
    KokkosGmgApply(const KokkosGmgApply&) = delete;
    KokkosGmgApply& operator=(const KokkosGmgApply&) = delete;

    virtual void apply(const double* r, double* z) = 0;

    [[nodiscard]] virtual int nlevels() const = 0;
};

// Build the hierarchy from the same face-coefficient pieces FaceCoeffOp takes. The
// fields must outlive the returned object, which reads them for the setup only.
//
// Triply periodic and red-black only: the ported V-cycle has no physical-BC handling
// and no Chebyshev smoother, so anything else throws rather than quietly solving a
// different problem.
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
);

} // namespace blockamr::bench
