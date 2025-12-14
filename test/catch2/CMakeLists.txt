# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

if(NeoN_WITH_MPI)
  add_library(NeoN_catch_main_mpi test_main_mpi.cpp mpiReporter.cpp mpiGlobals.cpp
                                  mpiSerialization.cpp)
  target_link_libraries(NeoN_catch_main_mpi PUBLIC Catch2::Catch2 cpptrace::cpptrace Kokkos::kokkos
                                                   NeoN MPI::MPI_CXX)
  target_link_libraries(NeoN_catch_main_mpi PRIVATE NeoN_warnings NeoN_options)
endif()

add_library(NeoN_catch_main test_main.cpp)
target_link_libraries(NeoN_catch_main PUBLIC Catch2::Catch2 cpptrace::cpptrace Kokkos::kokkos NeoN)
target_link_libraries(NeoN_catch_main PRIVATE NeoN_warnings NeoN_options)
