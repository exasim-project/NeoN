// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

namespace blockamr::solvers
{

using MLMG = amrex::MLMGT<amrex::MultiFab>;
using Dense = gko::matrix::Dense<double>;

} // namespace blockamr::solvers
