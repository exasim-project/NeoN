// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <memory>
#include <string>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// The optimised (kokkos_opt) V-cycle as a preconditioner apply, behind an interface
// that mentions neither Kokkos nor Ginkgo: the Kokkos kernels compile in
// blockamr_kokkos, the Ginkgo stack in blockamr_solvers (both
// CUDA_SEPARABLE_COMPILATION OFF; blockamr_kokkos is separate by history, not because
// of an RDC fence -- see CMakeLists.txt). So the cycle is an opaque handle over flat
// device vectors and precond.hpp wraps it in a gko::LinOp on the other side, with
// nothing shared but this header and two double pointers.
//
// Why: a V-cycle measured in isolation is not a preconditioner. What a caller sees is
// the SOLVE, where the cycle is one term next to the matrix-vector product, the Krylov
// algebra and the iteration count -- and the only way to compare it with MLMG and the
// native GMG on equal terms (bench_solvers.py runs all of them over the same operator).

namespace blockamr
{

// The V-cycle shape, matching GmgArgs (kokkosBench.hpp) and production's
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

    // The level storage type: "fp64", "fp32" or "bf16" (see GmgArgs::precision). The
    // flat vectors this class exchanges with the solver are fp64 regardless.
    std::string precision = "fp64";

    // The storage type of the COEFFICIENTS alone; empty = same as `precision`.
    // Narrowing these costs far fewer iterations than narrowing psi and rhs, because a
    // coefficient error perturbs only the preconditioner's operator, never the residual
    // CG stops on (kernels.hpp GmgGsCell has the measurements).
    std::string coeffPrecision;

    // On by default here, unlike in the bench: it cannot change the result at equal
    // depth (red-black smoothing is decomposition-independent), it keeps the coarse
    // levels from being one launch per tiny box, and the extra depth it unlocks helps
    // the iteration count.
    bool agglomerate = true;
    int aggGridSize = 32;

    // Target box size for level 0's own decomposition; 0 keeps the caller's boxes.
    // Trades one copy per apply for the halo traffic of the level holding most of the
    // cells.
    int aggLevel0Size = 0;

    // On by default for the same reason as agglomeration: it cannot change the result.
    // ux and lx are the SAME entries of a symmetric operator stored twice, so one face
    // fab per direction removes three of the nine arrays a colour sweep streams.
    // Symmetry is verified at setup and an asymmetric operator keeps the pair.
    bool shareCoeffs = true;

    // Homogeneous BCs per side (xlo, xhi, ylo, yhi, zlo, zhi): 0 periodic, 1 Dirichlet,
    // 2 Neumann -- la::BcArray's encoding, so a parsed spec passes straight through.
    std::array<int, 6> bc {};
};

// z = M^{-1} r on flat DEVICE vectors in the solver's cell ordering (MFIter order, i
// fastest within a valid box -- transfer.hpp's ordering).
//
// Two widths because the Krylov vectors have two: fp64 for the ordinary solvers, fp32
// for the inner solve of the mixed-precision refinement. Both are independent of the
// HIERARCHY's storage type (`precision` above); the scatter/gather converts.
class KokkosGmgApply
{
public:

    KokkosGmgApply() = default;
    virtual ~KokkosGmgApply() = default;
    KokkosGmgApply(const KokkosGmgApply&) = delete;
    KokkosGmgApply& operator=(const KokkosGmgApply&) = delete;

    virtual void apply(const double* r, double* z) = 0;
    virtual void apply(const float* r, float* z) = 0;

    [[nodiscard]] virtual int nlevels() const = 0;
};

// Build the hierarchy from the same face-coefficient pieces FaceCoeffOp takes. The
// fields are read at setup only and level 0 keeps its own converted copies, so later
// in-place writes by the caller are not seen -- a changed operator needs a rebuilt
// object.
//
// Red-black only: asking for Chebyshev throws rather than quietly smoothing
// differently. Boundary conditions -- periodic, homogeneous Dirichlet, homogeneous
// Neumann -- come through opts.bc, which must agree with the geometry's periodicity
// (la::parseBc enforces that at the solver boundary).
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

} // namespace blockamr
