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
set(NeoN_GINKGO_VERSION "1.11.0")
set(NeoN_GINKGO_TAG "49bff363a96b946f5e39bdbfbbb49a53e86050cf")
set(NeoN_CATCH2_VERSION "3.4.0")
set(NeoN_SPDLOG_VERSION "1.16.0")
set(NeoN_NANOBIND_VERSION "2.9.2")
set(NeoN_UMPIRE_TAG "18a808d1af81fed8823fcf12452d91c981f1bad1")
set(NeoN_FMT_VERSION "12.1.0")
