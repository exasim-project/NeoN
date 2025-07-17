// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_adapters.hpp"
#include "catch2/reporters/catch_reporter_registrars.hpp"
#include "Kokkos_Core.hpp"

#include "NeoN/core/mpi/environment.hpp"

#include "mpiReporter.hpp"
#include "mpiSerialization.hpp"

CATCH_REGISTER_REPORTER("mpi", MpiReporter);

int main(int argc, char* argv[])
{
    NeoN::mpi::MPIInit mpi(argc, argv);

    MPI_Comm_rank(COMM, &RANK);
    MPI_Comm_size(COMM, &COMM_SIZE);
    IS_ROOT = RANK == ROOT;

    // create a thread (on the root process) that serializes the IO
    bool threadShutdown = false;
    std::thread sequalizeIOThread {serializeIO, &threadShutdown};

    // Initialize Catch2
    Kokkos::initialize(argc, argv);

    // ensure any kokkos initialization output will appear first
    std::cout << std::flush;
    std::cerr << std::flush;
    MPI_Barrier(COMM);

    Catch::Session session;

    // Specify command line options
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    int result = session.run();
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, COMM);

    MPI_Barrier(COMM);
    threadShutdown = true;
    sequalizeIOThread.join();

    Kokkos::finalize();

    return result;
}
