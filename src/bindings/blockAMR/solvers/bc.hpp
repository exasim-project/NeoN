// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "bc_geom.hpp"

namespace blockamr::solvers
{

// Host-accessible (pinned) copy of a MultiFab. The coefficient fields arrive
// in the default arena — device memory in a GPU build — but the face-coeff stencil
// runs host-side on the ReferenceExecutor, so the (solve-constant) coefficients
// are staged to pinned memory once at operator construction.
std::shared_ptr<amrex::MultiFab> pinnedCopy(const amrex::MultiFab& src);

// BcArray, BcGhostFill and bcGhostFill live in bc_geom.hpp: the bench-side Kokkos
// V-cycle builds the same ghost fill and cannot link against bc.cpp.

BcArray
parseBc(const std::vector<std::string>& bc, const amrex::Geometry& geom, const std::string& who);

// fillDomainBcGhostsDevice / fillDomainBcGhostsHost also live in bc_geom.hpp, for the
// same reason: the bench-side V-cycle needs the AMReX fill as the reference its Kokkos
// twin is tested against, and cannot include this header (nanobind).

// Scatter ONLY the ghost-adjacent shell (outer 1-cell layer of each valid box)
// from the flat Ginkgo vector into the MultiFab (M3 3a). That shell is all that
// FillBoundary (periodic/internal) and the reflect domain-BC fill read to
// populate the face ghosts the fused stencil consults; the interior valid cells
// are read straight from the flat vector by faceCoeffStencilFusedDevice, so they
// need not be copied. Flat index matches scatter_device (box-by-box, i fastest).
void scatterShellDevice(const double* vec, amrex::MultiFab& mf);

} // namespace blockamr::solvers
