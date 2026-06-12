# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# Detect whether the local MPI installation provides MPI_THREAD_MULTIPLE at runtime.  The result is
# cached in NeoN_MPI_HAS_THREAD_MULTIPLE (BOOL). Called from CxxThirdParty.cmake after
# find_package(MPI).

include(CheckCXXSourceRuns)
include(CMakePushCheckState)

cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_runs(
  "#include <mpi.h>
   int main(int argc, char** argv)
   {
       int provided;
       MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
       MPI_Finalize();
       return (provided == MPI_THREAD_MULTIPLE) ? 0 : 1;
   }"
  NeoN_MPI_HAS_THREAD_MULTIPLE)
cmake_pop_check_state()

# Define the user-facing option here (not in CMakeLists.txt) so its initial default value reflects
# the detected runtime capability.  option() is a no-op when the variable is already in the CMake
# cache, so an explicit -DNeoN_ENABLE_MPI_WITH_THREAD_SUPPORT=OFF always wins.
option(NeoN_ENABLE_MPI_WITH_THREAD_SUPPORT "Enable MPI with threading support"
       ${NeoN_MPI_HAS_THREAD_MULTIPLE})

if(NeoN_MPI_HAS_THREAD_MULTIPLE)
  message(STATUS "NeoN: MPI_THREAD_MULTIPLE supported — "
                 "NeoN_ENABLE_MPI_WITH_THREAD_SUPPORT=${NeoN_ENABLE_MPI_WITH_THREAD_SUPPORT}")
else()
  message(
    WARNING "NeoN: this MPI installation does not provide MPI_THREAD_MULTIPLE.\n"
            "    Distributed benchmarks spawn a background IO thread that calls MPI;\n"
            "    without full thread support those runs may crash or produce wrong results.\n"
            "    Consider rebuilding MPI with thread support "
            "(e.g. OpenMPI: --enable-mpi-thread-multiple).")
endif()
