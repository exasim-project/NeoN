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

/* @brief One cell-centred field on ONE AMR level (one level of Python's CellField).
 *        The const and non-const operator* are a PAIR: shared_ptr does not propagate
 *        constness, so a lone const overload would hand out a writable MultiFab&.
 */
struct CellFieldLevel
{
    std::shared_ptr<amrex::MultiFab> mf;

    amrex::MultiFab& operator*() { return *mf; }
    const amrex::MultiFab& operator*() const { return *mf; }
};

/* @brief The three direction fields of one face-centred field, on one level. Exists to
 *        remove the ux/lx/uy/ly/uz/lz hazard: six adjacent parameters of identical type,
 *        where transposing two compiles and silently answers wrongly.
 */
struct FaceFieldLevel
{
    std::array<std::shared_ptr<amrex::MultiFab>, 3> dir {};

    amrex::MultiFab& operator[](int d) { return *dir[static_cast<std::size_t>(d)]; }
    const amrex::MultiFab& operator[](int d) const { return *dir[static_cast<std::size_t>(d)]; }
};

// A handle to a field owned elsewhere: an aliasing shared_ptr with an EMPTY owner, so no
// control block is allocated and no deleter ever runs.
inline std::shared_ptr<amrex::MultiFab> nonOwning(amrex::MultiFab& mf)
{
    return std::shared_ptr<amrex::MultiFab> {std::shared_ptr<amrex::MultiFab> {}, &mf};
}

} // namespace blockamr
