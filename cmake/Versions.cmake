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
set(NeoN_CATCH2_VERSION "3.4.0")
set(NeoN_SPDLOG_VERSION "1.16.0")
# Match the nanobind that pybFoam builds against (pip, currently 2.13.0). The _neon binding shares
# pybFoam's libnanobind.so at runtime (SONAME), so an ABI mismatch (e.g. the ndarray_create signature
# change after 2.9.2) makes _neon fail to resolve symbols from pybFoam's lib. Overridable so a
# downstream (NeoFOAM) can force a different version to track its own pybFoam.
if(NOT DEFINED NeoN_NANOBIND_VERSION)
  set(NeoN_NANOBIND_VERSION "2.13.0")
endif()
set(NeoN_UMPIRE_TAG "18a808d1af81fed8823fcf12452d91c981f1bad1")
set(NeoN_FMT_VERSION "12.1.0")
