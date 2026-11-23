// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <atomic>
#include <iostream>
#include <thread>

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_adapters.hpp"
#include "catch2/reporters/catch_reporter_registrars.hpp"
#include "Kokkos_Core.hpp"

#include "NeoN/core/initialization.hpp"
#include "NeoN/core/mpi/environment.hpp"

#include "mpiReporter.hpp"
#include "mpiSerialization.hpp"

CATCH_REGISTER_REPORTER("mpi", MpiReporter);

int main(int argc, char* argv[])
{
    NeoN::mpi::Init mpi(argc, argv);

    MPI_Comm_rank(COMM, &RANK);
    MPI_Comm_size(COMM, &COMM_SIZE);
    IS_ROOT = RANK == ROOT;

    // Create a thread (on the root process) that serializes the IO. That thread calls MPI
    // concurrently with the main thread, which is only defined behaviour at MPI_THREAD_MULTIPLE.
    // NeoN requests that level only when built with NF_REQUIRE_MPI_THREAD_SUPPORT, and CI
    // configures with NeoN_ENABLE_MPI_WITH_THREAD_SUPPORT=OFF, so query what the library actually
    // provided instead of assuming. Below MPI_THREAD_MULTIPLE the thread is not started and the
    // reporter skips the handshake: output from failing assertions on different ranks may then
    // interleave, which is much preferable to driving MPI from two threads at MPI_THREAD_SINGLE --
    // that is undefined behaviour and showed up as an intermittent segfault inside MPI_Irecv.
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Query_thread(&providedThreadLevel);
    IO_SERIALIZATION = providedThreadLevel == MPI_THREAD_MULTIPLE;

    std::atomic<bool> threadShutdown {false};
    std::thread sequalizeIOThread;
    if (IO_SERIALIZATION)
    {
        sequalizeIOThread = std::thread {serializeIO, &threadShutdown};
    }
    else if (IS_ROOT)
    {
        std::cout << "[NeoN] MPI thread level is below MPI_THREAD_MULTIPLE; test output from "
                     "different ranks is not serialized.\n";
    }

    // Initialize Catch2
    NeoN::initialize(argc, argv);

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
    if (sequalizeIOThread.joinable())
    {
        sequalizeIOThread.join();
    }

    NeoN::finalize();

    return result;
}
