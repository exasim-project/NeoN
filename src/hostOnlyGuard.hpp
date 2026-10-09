// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Force-included (-include) into every host-only translation unit, see NeoN_host in
// src/CMakeLists.txt and doc/splitCompilation.rst.
//
// Host-only TUs may be compiled by a different compiler and at a newer C++ standard than the
// Kokkos half of NeoN, so they must never see a Kokkos header. A missing include path is not a
// sufficient guard: in conda or system installs Kokkos sits in a default include directory. If
// you hit "attempt to use poisoned 'Kokkos'", a header reachable from this TU pulls in Kokkos.
// Either make that header Kokkos-free (use NeoN/core/portability.hpp for annotations) or move the
// TU to NeoN_SRCS.

#pragma once

#if !defined(NEON_HOST_ONLY_TU)
#error "hostOnlyGuard.hpp is only meant for host-only translation units"
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC poison Kokkos
#endif
