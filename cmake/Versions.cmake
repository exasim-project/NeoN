# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# Centralized version management for all third-party dependencies in NeoN

set(NeoN_KOKKOS_CHECKOUT_VERSION
    "4.7.01"
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
