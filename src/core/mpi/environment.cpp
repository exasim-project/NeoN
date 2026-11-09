// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/logging.hpp"

namespace NeoN::mpi
{

namespace
{
// MPI_COMM_NULL stands for "not set": MPI_COMM_WORLD is not a constant
// expression in every MPI implementation.
MPI_Comm DEFAULT_COMM = MPI_COMM_NULL;
}

MPI_Comm defaultCommunicator()
{
    return DEFAULT_COMM == MPI_COMM_NULL ? MPI_COMM_WORLD : DEFAULT_COMM;
}

void setDefaultCommunicator(MPI_Comm comm)
{
    DEFAULT_COMM = comm == MPI_COMM_WORLD ? MPI_COMM_NULL : comm;
    Logging::updateRankPolicy();
}

bool hasDefaultCommunicator() { return DEFAULT_COMM != MPI_COMM_NULL; }

}
