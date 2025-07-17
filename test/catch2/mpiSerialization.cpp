// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "mpiSerialization.hpp"

#include <thread>
#include <vector>

#include "mpiGlobals.hpp"

void serializeIO(volatile bool* threadShutdown)
{
    if (!IS_ROOT)
    {
        return;
    }

    std::vector<MPI_Request> req(static_cast<std::size_t>(COMM_SIZE));
    bool recvBuffer;
    for (int i = 0; i < COMM_SIZE; ++i)
    {
        MPI_Irecv(
            &recvBuffer,
            1,
            MPI_CXX_BOOL,
            i,
            SERIALIZATION_TAG,
            COMM,
            &req[static_cast<std::size_t>(i)]
        );
    }

    while (!*threadShutdown)
    {
        // completed will be the index within req of a request that has finished. This is
        // equivalent to rank of the process who wants to print
        int completed = MPI_UNDEFINED;
        int flag;
        while (completed == MPI_UNDEFINED && !*threadShutdown)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            MPI_Testany(
                static_cast<int>(req.size()), req.data(), &completed, &flag, MPI_STATUS_IGNORE
            );
        }

        if (*threadShutdown)
        {
            break;
        }

        // allow process to print
        const bool allowPrint = true;
        MPI_Send(&allowPrint, 1, MPI_CXX_BOOL, completed, SERIALIZATION_TAG, COMM);

        // wait until process has finished printing
        bool remoteHasFinished = false;
        MPI_Recv(
            &remoteHasFinished,
            1,
            MPI_CXX_BOOL,
            completed,
            SERIALIZATION_TAG,
            COMM,
            MPI_STATUS_IGNORE
        );

        // re-activate request for process
        MPI_Irecv(
            &recvBuffer,
            1,
            MPI_CXX_BOOL,
            completed,
            SERIALIZATION_TAG,
            COMM,
            &req[static_cast<std::size_t>(completed)]
        );
    }
}
