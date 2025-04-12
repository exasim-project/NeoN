// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#ifdef NF_WITH_MPI_SUPPORT
#include <mpi.h>
#endif

#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/info.hpp"


namespace NeoFOAM
{

#ifdef NF_WITH_MPI_SUPPORT

namespace mpi
{

/**
 * @struct MPIInit
 * @brief A RAII class to manage MPI initialization and finalization with thread support.
 */
struct MPIInit
{
    /**
     * @brief Initializes the MPI environment, ensuring thread support.
     *
     * @param argc Reference to the argument count.
     * @param argv Reference to the argument vector.
     */
    MPIInit(int argc, char** argv)
    {
#ifdef NF_REQUIRE_MPI_THREAD_SUPPORT
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        NF_ASSERT(
            provided == MPI_THREAD_MULTIPLE, "The MPI library does not have full thread support"
        );
#else
        MPI_Init(&argc, &argv);
#endif
    }

    /**
     * @brief Destroy the MPIInit object.
     */
    ~MPIInit() { MPI_Finalize(); }
};


/**
 * @class MPIEnvironment
 * @brief Manages the MPI environment, including rank and rank size information.
 */
class MPIEnvironment
{
public:

    /**
     * @brief Initializes the MPI environment using a parsed communicator group.
     *
     * @param commGroup The communicator group, default is MPI_COMM_WORLD.
     */
    MPIEnvironment(MPI_Comm commGroup = MPI_COMM_WORLD) : communicator(commGroup)
    {
        updateRankData();
    }

    /**
     * @brief Finalizes the MPI environment.
     */
    ~MPIEnvironment() = default;

    /**
     * @brief Returns the number of ranks.
     *
     * @return The number of ranks.
     */
    size_t sizeRank() const { return static_cast<size_t>(mpi_size); }

    /**
     * @brief Returns the rank of the current process.
     *
     * @return The rank of the current process.
     */
    size_t rank() const { return static_cast<size_t>(mpi_rank); }

    /**
     * @brief Returns the communicator.
     *
     * @return The communicator.
     */
    MPI_Comm comm() const { return communicator; }

private:

    MPI_Comm communicator {MPI_COMM_NULL}; // MPI communicator
    int mpi_rank {-1};                     // Index of this rank
    int mpi_size {-1};                     // Number of ranks in this communicator group.

    /**
     * @brief Updates the rank data, based on the communicator.
     */
    void updateRankData()
    {
        NF_ASSERT(communicator != MPI_COMM_NULL, "Invalid communicator, is null.");
        MPI_Comm_rank(communicator, &mpi_rank);
        MPI_Comm_size(communicator, &mpi_size);
    }
};

} // namespace mpi

#endif

} // namespace NeoFOAM
