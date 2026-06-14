# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# Centralized version management for all third-party dependencies in NeoN

set(NeoN_KOKKOS_CHECKOUT_VERSION
    "5.0.2"
    CACHE STRING "Use specific version of Kokkos")
mark_as_advanced(NeoN_KOKKOS_CHECKOUT_VERSION)

set(NeoN_CPPTRACE_VERSION "0.7.3")
set(NeoN_ADIOS2_VERSION "2.10.2")
set(NeoN_SUNDIALS_VERSION "7.5.0")
set(NeoN_JSON_VERSION "3.11.3")
set(NeoN_GINKGO_VERSION "2.0.0")
set(NeoN_GINKGO_TAG "6a3abf8c920228006f3b28bc3bf04fc7a5f6aee0")

# Opt-in: build against Ginkgo PR #2000 "Add a public workspace" (ginkgo-project/ginkgo#2000, branch
# feat/public-workspace) to enable reuse of a solver's scratch Workspace across generate() calls.
# The PR is unmerged/review-blocked and its API is in flux, so this is OFF by default; the C++ path
# that uses it is guarded by the NEON_GINKGO_PUBLIC_WORKSPACE compile macro. Turning this ON re-pins
# Ginkgo to the PR head commit (which sits on a newer Ginkgo develop than the stable 2.0.0 pin) and
# forces a Ginkgo rebuild. Remove this block and bump NeoN_GINKGO_TAG once the PR is merged.
option(NeoN_GINKGO_PUBLIC_WORKSPACE
       "Pin Ginkgo to PR #2000 (public workspace) and enable solver workspace reuse" OFF)
if(NeoN_GINKGO_PUBLIC_WORKSPACE)
  set(NeoN_GINKGO_TAG "5c35072cefd0e3c3d9261f869f378730742bf721")
  message(STATUS "NeoN: Ginkgo pinned to PR #2000 (public workspace) @ ${NeoN_GINKGO_TAG}")
endif()
set(NeoN_CATCH2_VERSION "3.4.0")
set(NeoN_SPDLOG_VERSION "1.16.0")
set(NeoN_NANOBIND_VERSION "2.9.2")
set(NeoN_UMPIRE_TAG "18a808d1af81fed8823fcf12452d91c981f1bad1")
set(NeoN_FMT_VERSION "12.1.0")
