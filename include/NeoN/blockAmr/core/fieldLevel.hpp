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
 * The `Level` suffix is load-bearing. Python's CellField
 * (src/blockAmr/python/blockamr/field.py) is a MULTI-level container indexed by
 * level; this is a single level of one, which is the granularity everything in
 * the linear algebra works at. Python's own per-level type is _FaceFieldLevel.
 *
 * The const and non-const operator* are a PAIR on purpose: shared_ptr does not
 * propagate constness, so a lone `operator*() const` would hand out a writable
 * amrex::MultiFab& through a const handle. With the pair, a
 * `const CellFieldLevel&` parameter is a read-only field and a by-value or
 * non-const one is a writable one -- so a by-value parameter is NOT a
 * const-preserving copy, it is write access.
 */
struct CellFieldLevel
{
    std::shared_ptr<amrex::MultiFab> mf;

    amrex::MultiFab& operator*() { return *mf; }
    const amrex::MultiFab& operator*() const { return *mf; }
};

/* @brief The three direction fields of one face-centred field, on one level.
 *
 * This is what removes the ux/lx/uy/ly/uz/lz hazard: six adjacent parameters of
 * identical type, where transposing two of them compiles cleanly and silently
 * produces a wrong answer. Same const/non-const accessor pair as
 * CellFieldLevel, for the same reason.
 */
struct FaceFieldLevel
{
    std::array<std::shared_ptr<amrex::MultiFab>, 3> dir {};

    amrex::MultiFab& operator[](int d) { return *dir[static_cast<std::size_t>(d)]; }
    const amrex::MultiFab& operator[](int d) const { return *dir[static_cast<std::size_t>(d)]; }
};

// A handle to a field whose lifetime is owned elsewhere -- the aliasing
// shared_ptr with an EMPTY owner, so no control block is allocated and no
// deleter ever runs. The entry points that receive a bare amrex::MultiFab&
// (Python owns the field; the solver facades pass raw pointers) build their
// handles through this rather than taking ownership of a caller's field.
inline std::shared_ptr<amrex::MultiFab> nonOwning(amrex::MultiFab& mf)
{
    return std::shared_ptr<amrex::MultiFab> {std::shared_ptr<amrex::MultiFab> {}, &mf};
}

} // namespace blockamr
