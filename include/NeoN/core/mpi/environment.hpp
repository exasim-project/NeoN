// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef NF_WITH_MPI_SUPPORT
#include <mpi.h>
#endif

#include "NeoN/core/error.hpp"
#include "NeoN/core/info.hpp"


namespace NeoN
{

#ifdef NF_WITH_MPI_SUPPORT

namespace mpi
{

/**
 * @struct Init
 * @brief A RAII class to manage MPI initialization and finalization with thread support.
 */
struct Init
{
    /**
     * @brief Initializes the MPI environment, ensuring thread support.
     *
     * @param argc Reference to the argument count.
     * @param argv Reference to the argument vector.
     */
    Init(int argc, char** argv)
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
     * @brief Destroy the Init object.
     */
    ~Init() { MPI_Finalize(); }
};


/**
 * @class Environment
 * @brief Manages the MPI environment, including rank and rank size information.
 */
class Environment
{
public:

    /**
     * @brief Initializes the MPI environment using a parsed communicator group.
     *
     * @param commGroup The communicator group, default is MPI_COMM_WORLD.
     */
    Environment(MPI_Comm commGroup = MPI_COMM_WORLD) : communicator(commGroup)
    {
        MPI_Initialized(&mpiInitialized);
        updateRankData();
    }

    /**
     * @brief Finalizes the MPI environment.
     */
    ~Environment() = default;

    /**
     * @brief returns if
     *
     * @return The number of ranks.
     */
    bool isInitialized() const { return mpiInitialized == 1; }

    /**
     * @brief Returns the number of ranks.
     *
     * @return The number of ranks.
     */
    size_t sizeRank() const { return static_cast<size_t>(mpiSize); }

    /**
     * @brief Returns the rank of the current process.
     *
     * @return The rank of the current process.
     */
    size_t rank() const { return static_cast<size_t>(mpiRank); }

    /**
     * @brief Returns the communicator.
     *
     * @return The communicator.
     */
    MPI_Comm comm() const { return communicator; }

private:

    MPI_Comm communicator {MPI_COMM_NULL}; // MPI communicator
    int mpiInitialized {0};
    int mpiRank {-1}; // Index of this rank
    int mpiSize {-1}; // Number of ranks in this communicator group.

    /**
     * @brief Updates the rank data, based on the communicator.
     */
    void updateRankData()
    {
        if (mpiInitialized)
        {
            MPI_Comm_rank(communicator, &mpiRank);
            MPI_Comm_size(communicator, &mpiSize);
        }
    }
};

} // namespace mpi

#endif

} // namespace NeoN
