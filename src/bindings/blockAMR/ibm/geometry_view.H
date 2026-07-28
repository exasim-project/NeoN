// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// PROVISIONAL — shipped by B28, frozen by B29 (review.md §4, Q29a/Q29b).
//
// Q29(b) rules the method-agnostic geometry layer (design §2.1) into ONE packed
// eight-component MultiFab behind a device-copyable POD view:
//
//     comp 0      sdf          f64, > 0 in the fluid
//     comp 1..3   normal       unit into-fluid normal at the cell centre
//     comp 4..6   wall_point   body intercept x - sdf * n
//     comp 7      patch        owning body id, carried as f64
//
// B28 reads exactly one of those components (`sdf`, in `classifyDefault`) but
// takes the whole container, because api §6.2's canonical spelling is
// `classifyDefault(CellTypeFab&, const IbmGeometryFab&, const Geometry&)` and
// Q29(a) forbids a temporary decoupled signature that B29 would then have to
// change. B29 owns making the mesh produce this fab and freezing the layout.

#include <AMReX_Array4.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <stdexcept>
#include <string>

namespace ibm
{

//! Component offsets of the packed geometry fab. Q29(b)'s layout, in one place.
inline constexpr int GEOM_SDF = 0;
inline constexpr int GEOM_NORMAL = 1;
inline constexpr int GEOM_WALL_POINT = 4;
inline constexpr int GEOM_PATCH = 7;
inline constexpr int GEOM_NCOMP = 8;

//! The host-side container (api §6.2's name for it).
using IbmGeometryFab = amrex::MultiFab;

//! The device-copyable POD a kernel captures by value (design §1.3/§2.4).
struct IbmGeometryView
{
    amrex::Array4<const amrex::Real> a;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real sdf(int i, int j, int k) const noexcept
    {
        return a(i, j, k, GEOM_SDF);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    normal(int i, int j, int k, int d) const noexcept
    {
        return a(i, j, k, GEOM_NORMAL + d);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    wallPoint(int i, int j, int k, int d) const noexcept
    {
        return a(i, j, k, GEOM_WALL_POINT + d);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int patch(int i, int j, int k) const noexcept
    {
        return static_cast<int>(a(i, j, k, GEOM_PATCH));
    }
};

//! One box of the packed fab, as the device view.
inline IbmGeometryView makeGeometryView(const IbmGeometryFab& g, const amrex::MFIter& mfi)
{
    return IbmGeometryView {g.const_array(mfi)};
}

//! Host guard: the packed layout is a contract, and a wrong width is a sentence.
inline void requireGeometryLayout(const IbmGeometryFab& g, const char* who)
{
    if (g.nComp() != GEOM_NCOMP)
        throw std::runtime_error(
            std::string(who) + ": the immersed-body geometry is one packed MultiFab of "
            + std::to_string(GEOM_NCOMP)
            + " components (sdf, normal x3, wall_point x3, patch), but this one has "
            + std::to_string(g.nComp()) + "; rebuild the geometry"
        );
}

} // namespace ibm
