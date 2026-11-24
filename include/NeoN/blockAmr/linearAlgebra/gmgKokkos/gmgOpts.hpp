// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <string>

// The Kokkos V-cycle's shape as ONE struct, in production: the preconditioner (precond.cpp), the
// cycle itself (vcycle.hpp) and the bench under bench/ all take these knobs from here, so the
// cycle no longer reaches into bench/ for them. Free of Kokkos, AMReX and nanobind headers, like
// the bench header that includes it.

namespace blockamr
{

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

} // namespace blockamr
