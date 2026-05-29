// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "NeoN/core/primitives/label.hpp"

#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN
{

/**
 * @struct CommunicationPattern
 * @brief Collects all data required for a distributed halo exchange.
 *
 * Built once per mesh partition by `computeCommunicationPattern` and then
 * reused for every field synchronisation step.
 */
struct CommunicationPattern
{
    /** @brief Number of faces sent to each rank (indexed by rank).
     *
     *  `sendCounts[r]` is the number of boundary faces this rank sends to rank
     *  `r`.  The extra element at index `nRanks` holds the total send count.
     */
    std::vector<int> sendCounts;

    /** @brief Global cell indices received from all neighbour ranks.
     *
     *  Populated by an `MPI_Alltoallv` exchange.  Each entry is the global
     *  index of the cell whose data must be written into the local halo after
     *  communication completes.
     */
    std::vector<int> recvIdx;

    /** @brief Maps received halo entries to local boundary face indices.
     *
     *  Currently unused / reserved for future mapping optimisations.
     */
    std::vector<localIdx> boundaryMapVector;

    /** @brief MPI environment captured at pattern-construction time. */
    mpi::Environment env;

    /** @brief Set by `computeCommunicationPattern` after an `MPI_Allreduce`
     *  so every rank in the job agrees on the dispatch decision.  True iff
     *  the mesh used to build this pattern was a partition (some rank in
     *  the job has processor faces), false for non-partitioned meshes
     *  (e.g. a per-rank full copy of the global mesh used for sanity checks).
     */
    bool isPartitioned = false;

    /** @brief Whether the owning LinearSystem participates in a distributed solve.
     *
     *  Authoritative dispatch question for solvers and post-assembly functors.
     *  Crucially, NOT equivalent to:
     *   - "this rank has processor faces" — would deadlock against peers when
     *     an irregular Scotch partition gives this rank an island bounded only
     *     by physical patches;
     *   - "the job runs multi-rank" — a multi-rank job can carry a
     *     non-partitioned LinearSystem (e.g. a sanity-check copy of the global
     *     mesh held by every rank) that must be solved locally.
     */
    bool isDistributed() const { return isPartitioned; }
};

/**
 * @brief Computes the CommunicationPattern for a distributed mesh partition.
 *
 * For each processor boundary patch the function collects the global cell
 * indices of the owner cells and exchanges them with neighbouring ranks via
 * `MPI_Alltoall` / `MPI_Alltoallv` so that every rank knows which global
 * cells it will receive during a halo exchange.
 *
 * @param mesh  The local mesh partition.  Must be a distributed mesh
 *              (i.e. `mesh.boundaryMesh().isDistributed()` returns `true`);
 *              an empty pattern is returned otherwise.
 * @return      A fully populated `CommunicationPattern` ready for use in
 *              field synchronisation.
 */
CommunicationPattern computeCommunicationPattern(const UnstructuredMesh& mesh);

} // namespace NeoN

#endif // NF_WITH_MPI_SUPPORT
