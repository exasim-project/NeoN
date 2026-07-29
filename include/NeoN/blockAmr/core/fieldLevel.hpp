// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MultiFab.H>

#include <array>
#include <cstddef>
#include <memory>

namespace blockamr
{

/* @brief One cell-centred field on ONE AMR level.
 *
 * The `Level` suffix is load-bearing: Python's CellField is a MULTI-level
 * container indexed by level, this is a single level of one -- the granularity
 * everything in the linear algebra works at.
 *
 * The const and non-const operator* are a PAIR on purpose: shared_ptr does not
 * propagate constness, so a lone `operator*() const` would hand out a writable
 * amrex::MultiFab& through a const handle. With the pair, a by-value parameter is
 * NOT a const-preserving copy, it is write access.
 */
struct CellFieldLevel
{
    std::shared_ptr<amrex::MultiFab> mf;

    amrex::MultiFab& operator*() { return *mf; }
    const amrex::MultiFab& operator*() const { return *mf; }
};

/* @brief The three direction fields of one face-centred field, on one level.
 *
 * Removes the ux/lx/uy/ly/uz/lz hazard: six adjacent parameters of identical type,
 * where transposing two compiles cleanly and silently answers wrongly. Same
 * accessor pair as CellFieldLevel, for the same reason.
 */
struct FaceFieldLevel
{
    std::array<std::shared_ptr<amrex::MultiFab>, 3> dir {};

    amrex::MultiFab& operator[](int d) { return *dir[static_cast<std::size_t>(d)]; }
    const amrex::MultiFab& operator[](int d) const { return *dir[static_cast<std::size_t>(d)]; }
};

// A handle to a field owned elsewhere -- the aliasing shared_ptr with an EMPTY
// owner, so no control block is allocated and no deleter ever runs. Entry points
// receiving a bare amrex::MultiFab& build their handles through this rather than
// taking ownership of a caller's field.
inline std::shared_ptr<amrex::MultiFab> nonOwning(amrex::MultiFab& mf)
{
    return std::shared_ptr<amrex::MultiFab> {std::shared_ptr<amrex::MultiFab> {}, &mf};
}

} // namespace blockamr
