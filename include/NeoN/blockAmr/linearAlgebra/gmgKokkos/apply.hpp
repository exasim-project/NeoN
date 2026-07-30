// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/gmgOpts.hpp"

// The optimised (kokkos_opt) V-cycle as a preconditioner apply, behind an interface that
// mentions neither Kokkos nor Ginkgo -- the two stacks compile in separate object libraries.
// Why an opaque handle, and why a preconditioner: report/blockamr-gmg-notes.md#kokkos-handle.

namespace blockamr
{

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
