// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>

namespace blockamr::solvers
{

// Assemble the face-coefficient matrix into a CSR on `exec`. SINGLE-BOX
// periodic only (matches the benchmark meshes): neighbour column indices wrap
// around the domain, and the row/column order is the same idx(i,j,k) =
// (k*nj + j)*ni + i used by the gather/scatter pack. This is the assembled
// counterpart of FaceCoeffOp, for measuring the matrix-free advantage.
std::shared_ptr<gko::matrix::Csr<double, int>> assembleFaceCoeffCsr(
    std::shared_ptr<const gko::Executor> exec,
    const amrex::Geometry& geom,
    const amrex::MultiFab& alpha,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz
);

} // namespace blockamr::solvers
