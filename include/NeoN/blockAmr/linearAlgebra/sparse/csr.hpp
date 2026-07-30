// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/meshLevel.hpp"

namespace blockamr::la
{

// Assemble the face-coefficient matrix into a CSR on `exec`: the assembled counterpart of
// FaceCoeffOp. SINGLE-BOX only, in the gather/scatter's row/column order
// idx(i,j,k) = (k*nj + j)*ni + i. `lower` is the STORED low side (aliasing upper if symmetric).

// The CSR half of the BC contract (the operator's half: operators/laplacian.hpp): a PERIODIC
// side keeps the modular-wraparound column; a Dirichlet/Neumann side DROPS it and the domain
// face's coefficient lands on the DIAGONAL as sign*aFace. Homogeneous BCs only.
std::shared_ptr<gko::matrix::Csr<double, int>> assembleFaceCoeffCsr(
    std::shared_ptr<const gko::Executor> exec,
    const MeshLevel& mesh,
    const CellFieldLevel& alpha,
    const FaceFieldLevel& upper,
    const FaceFieldLevel& lower,
    const BcArray& bc
);

} // namespace blockamr::la
