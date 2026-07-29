// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H> // fillHalo

namespace blockamr
{

/* @brief The AMReX layout triple -- BoxArray, DistributionMapping, Geometry --
 *        travelling as ONE argument.
 *
 * The `Level` suffix matches CellFieldLevel/FaceFieldLevel (core/fieldLevel.hpp):
 * this is the layout of ONE AMR level, the granularity everything in the linear
 * algebra works at. Python's `blockamr.Mesh`/`AmrMesh` are the multi-level
 * containers, and the name is already taken there.
 *
 * HELD BY VALUE, which is what makes it safe as a MEMBER as well as a parameter:
 * ba and dm are refcounted handles onto shared, immutable layout data and Geometry
 * is a plain value, so a copy is cheap and a member cannot dangle.
 *
 * The executor is deliberately NOT here. la routes it through the MATRIX
 * (IsMatrix::executor()) so an operator launches where its coefficient fields
 * live; a second source for it is the failure this grouping exists to prevent.
 */
struct MeshLevel
{
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    amrex::Geometry geom;

    // By value: a reference into a temporary geometry is the bug this would
    // otherwise invite in device code.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx() const { return geom.CellSizeArray(); }

    amrex::Periodicity periodicity() const { return geom.periodicity(); }

    // The halo a coefficient field needs before a face average reads its
    // neighbour. Internal and periodic ghosts only -- a physical domain face has no
    // second cell and is the BC's business.
    void fillHalo(amrex::MultiFab& mf) const { mf.FillBoundary(periodicity()); }
};

} // namespace blockamr
