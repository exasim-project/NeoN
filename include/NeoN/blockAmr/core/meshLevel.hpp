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

/* @brief The AMReX layout triple (BoxArray, DistributionMapping, Geometry) of ONE AMR
 *        level, as one argument. Held BY VALUE, so safe as a member; the executor is
 *        deliberately NOT here -- la routes it through the MATRIX instead.
 */
struct MeshLevel
{
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    amrex::Geometry geom;

    // By value: a reference into a temporary geometry is the bug this avoids.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx() const { return geom.CellSizeArray(); }

    amrex::Periodicity periodicity() const { return geom.periodicity(); }

    // The halo a coefficient field needs before a face average reads its neighbour.
    // Internal and periodic ghosts only -- a domain face is the BC's business.
    void fillHalo(amrex::MultiFab& mf) const { mf.FillBoundary(periodicity()); }
};

} // namespace blockamr
