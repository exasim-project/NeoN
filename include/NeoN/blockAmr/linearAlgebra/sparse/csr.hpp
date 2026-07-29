// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>

#include "NeoN/blockAmr/core/bc.hpp"

namespace blockamr::la
{

// Assemble the face-coefficient matrix into a CSR on `exec`: the assembled
// counterpart of FaceCoeffOp, for measuring the matrix-free advantage. SINGLE-BOX
// only (the benchmark meshes), in the gather/scatter's own row/column order
// idx(i,j,k) = (k*nj + j)*ni + i.
//
// The CSR half of the boundary-condition contract (the operator's half is in
// operators/laplacian.hpp): a PERIODIC side keeps the modular-wraparound neighbour
// column; on a Dirichlet/Neumann side the reflected ghost (core/bc.hpp) makes the
// outside value sign*pC, so that column is DROPPED and the domain face's
// coefficient lands on the DIAGONAL as sign*aFace (sign = -1 / +1) — such rows
// carry fewer than 7 entries. Homogeneous only: inhomogeneous `bc_data` is an rhs
// fold, not a matrix one, and FaceCoeffCsrSolver refuses it.
std::shared_ptr<gko::matrix::Csr<double, int>> assembleFaceCoeffCsr(
    std::shared_ptr<const gko::Executor> exec,
    const amrex::Geometry& geom,
    const amrex::MultiFab& alpha,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const BcArray& bc
);

} // namespace blockamr::la
