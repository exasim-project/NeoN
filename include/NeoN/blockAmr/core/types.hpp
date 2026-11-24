// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>

namespace blockamr::la
{

using MLMG = amrex::MLMGT<amrex::MultiFab>;

} // namespace blockamr::la
