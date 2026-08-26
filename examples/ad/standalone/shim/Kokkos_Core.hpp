// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// Minimal stand-in for Kokkos_Core.hpp, providing only the facilities the AD
// primitives use. This lets examples/ad/standalone be built and the numerics
// verified with a plain host compiler and no Kokkos installation.
//
// It is never on the include path of a normal NeoN build: dual.hpp includes
// <Kokkos_Core.hpp> exactly like every other NeoN header, and CMake resolves it
// to the real thing. This directory is prepended only by the example's own
// build line.

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#include <cmath>

namespace Kokkos
{
using std::exp;
using std::log;
using std::sqrt;
}
