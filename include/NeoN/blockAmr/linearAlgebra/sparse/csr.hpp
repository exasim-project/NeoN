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

// Assemble the face-coefficient matrix into a CSR on `exec`. SINGLE-BOX only
// (matches the benchmark meshes); the row/column order is the same idx(i,j,k) =
// (k*nj + j)*ni + i used by the gather/scatter pack. This is the assembled
// counterpart of FaceCoeffOp, for measuring the matrix-free advantage.
//
// `bc` carries the same homogeneous domain conditions FaceCoeffOp folds by ghost
// reflection (core/bc.hpp): a PERIODIC side keeps the modular-wraparound
// neighbour column, while on a Dirichlet/Neumann side the reflected ghost makes
// the outside neighbour's value sign*pC (sign = -1 / +1), so that face has no
// column at all and its coefficient lands on the DIAGONAL as sign*aFace. Rows on
// a non-periodic boundary therefore carry fewer than 7 entries.
//
// Homogeneous only: inhomogeneous `bc_data` is an rhs fold, not a matrix one, and
// FaceCoeffCsrSolver refuses it.
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
