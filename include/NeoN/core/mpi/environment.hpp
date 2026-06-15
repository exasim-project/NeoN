// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef NF_WITH_MPI_SUPPORT
#include <mpi.h>
#endif

#include <cstdlib>

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
        gpuAwareMpi_ = probeGpuAwareMpi();
        if (std::getenv("NEON_FORCE_HOST_BUFFER") != nullptr) gpuAwareMpi_ = false;
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

    /**
     * @brief Returns whether GPU-aware MPI is enabled (default: true).
     *
     * Set the environment variable NEON_FORCE_HOST_BUFFER to disable GPU-aware MPI
     * and force host-side staging buffers for all communication.
     */
    bool gpuAwareMpi() const { return gpuAwareMpi_; }

    /**
     * @brief Sets whether GPU-aware MPI is enabled at runtime.
     */
    bool& gpuAwareMpi() { return gpuAwareMpi_; }

    /**
     * @brief Returns the MPI tag upper bound (MPI_TAG_UB), cached for the process lifetime.
     *
     * MPI_Comm_get_attr is a local (non-collective) query. The result is cached once via a
     * thread-safe (C++11) function-local static. Falls back to 32767 (the MPI-1+ guaranteed
     * minimum) when the attribute is absent (found==0) or the pointer is null.
     */
    int tagUpperBound() const
    {
        static const int cachedTagUb = []() -> int
        {
            void* attrVal = nullptr;
            int found = 0;
            MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &attrVal, &found);
            if (found && attrVal != nullptr) return *static_cast<int*>(attrVal);
            return 32767; // MPI-1+ guaranteed minimum (Pitfall 3: honour the found flag)
        }();
        return cachedTagUb;
    }

    /**
     * @brief Runtime-probed GPU-aware MPI capability, cached once for the process lifetime.
     *
     * Probes MPIX_Query_cuda_support() under #ifdef MPIX_CUDA_AWARE_SUPPORT. When the symbol
     * is absent, defaults to false (host staging) — never pass device pointers to an unprobed
     * MPI. NEON_FORCE_HOST_BUFFER always overrides this (the ctor applies the override after
     * seeding gpuAwareMpi_ from this probe).
     *
     * CUDA is the only backend wired now. Drop-in slots for HIP/ROCm and SYCL/Level-Zero are
     * left as comments — each absent symbol defaults to host staging.
     */
    static bool probeGpuAwareMpi()
    {
        static const bool probed = []() -> bool
        {
#ifdef MPIX_CUDA_AWARE_SUPPORT
            return MPIX_Query_cuda_support() == 1;
            // Drop-in slots for other GPU backends (not wired in Phase 9):
            //   HIP/ROCm:  #elif defined(MPIX_ROCM_AWARE_SUPPORT) return MPIX_Query_rocm_support()
            //   == 1; SYCL/L0:   #elif defined(MPIX_ZE_AWARE_SUPPORT)   return
            //   MPIX_Query_ze_support()   == 1;
            // Each resolves the same single gpuAwareMpi() capability per the active Kokkos
            // GPUExecutor backend; each absent symbol defaults to host staging.
#else
            return false; // symbol absent => default to host staging (safe; never pass device ptrs
                          // to an unprobed MPI)
#endif
        }();
        return probed;
    }

private:

    MPI_Comm communicator {MPI_COMM_NULL}; // MPI communicator
    int mpiInitialized {0};
    int mpiRank {-1}; // Index of this rank
    int mpiSize {-1}; // Number of ranks in this communicator group.
    bool gpuAwareMpi_ {true};

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
