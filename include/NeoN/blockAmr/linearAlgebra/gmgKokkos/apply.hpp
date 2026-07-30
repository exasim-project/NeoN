// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <memory>
#include <string>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// The optimised (kokkos_opt) V-cycle as a preconditioner apply, behind an interface that
// mentions neither Kokkos nor Ginkgo -- the two stacks compile in separate object libraries.
// Why an opaque handle, and why a preconditioner: report/blockamr-gmg-notes.md#kokkos-handle.

namespace blockamr
{

// The V-cycle shape, matching GmgArgs (kokkosBench.hpp) and production's gmg_* options.
struct KokkosGmgOpts
{
    int cycles = 1;
    int preSweeps = 2;
    int postSweeps = 2;
    int coarsestSweeps = 8;
    int maxLevels = 0; // 0 = coarsen as far as the grid allows
    int minBottom = 2;
    double omega = 1.0;

    // The level storage type: "fp64", "fp32" or "bf16"; the flat vectors stay fp64 regardless.
    std::string precision = "fp64";

    // The COEFFICIENTS' storage type alone; empty = same as `precision`. Narrowing these costs
    // far fewer iterations than narrowing the fields: report/blockamr-precision-measurements.md.
    std::string coeffPrecision;

    // On by default: it cannot change the result at equal depth (notes#agglomeration).
    bool agglomerate = true;
    int aggGridSize = 32;

    // Target box size for level 0's own decomposition; 0 keeps the caller's boxes.
    int aggLevel0Size = 0;

    // On by default; it cannot change the result and drops 3 of 9 arrays (notes#share-coeffs).
    bool shareCoeffs = true;

    // Homogeneous BCs per side, la::BcArray's encoding: 0 periodic, 1 Dirichlet, 2 Neumann.
    std::array<int, 6> bc {};
};

// z = M^{-1} r on flat DEVICE vectors in the solver's cell ordering (transfer.hpp). Two widths
// because the Krylov vectors have two; both are independent of the HIERARCHY's storage type.
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

// Build the hierarchy from the same face-coefficient pieces FaceCoeffOp takes. The fields are
// read at SETUP only and level 0 keeps its own copies, so later caller writes go unseen -- a
// changed operator needs a rebuilt object. Red-black only; homogeneous BCs via opts.bc.
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
