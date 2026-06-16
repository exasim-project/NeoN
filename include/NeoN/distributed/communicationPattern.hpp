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

    /** @brief Maps the k-th rank-grouped recv entry to its local proc-boundary face index.
     *
     *  `boundaryMapVector[k]` is the local processor-boundary face index (0-based within the
     *  proc-boundary block, range [0, nProcBoundaryFaces)) of the k-th entry in the rank-grouped
     *  receive buffer (`recvRankGrouped` layout with `rdispl[r]` offsets). Populated by
     *  `computeCommunicationPattern` alongside `recvIdx` as the inverse permutation of the
     *  recvIdx scatter walk. Consumed by the unified halo-exchange primitive to scatter received
     *  data into a field's proc-boundary tail: `value_[procFaceStart + boundaryMapVector[k]]`,
     *  where `procFaceStart = mesh.nBoundaryFaces()` (physical-boundary count,
     *  i.e. the first proc-face index inside BoundaryData::value_).
     */
    std::vector<localIdx> boundaryMapVector;

    /** @brief MPI environment captured at pattern-construction time. */
    mpi::Environment env;

    /** @brief Row-sort permutation for the off-diagonal (processor-face) matrix entries.
     *
     *  `offDiagRowSortPerm[i]` is the processor-face (assembly / proc-face order) index whose
     *  off-diagonal value belongs at sorted position `i`. Computed once when the off-diagonal
     *  sparsity is created (see `createEmptyLinearSystem`) so the non-local COO handed to Ginkgo
     *  can be row-sorted without re-sorting on every matrix build. Ginkgo's CUDA `Coo::apply2`
     *  (the non-local/halo apply) requires row-sorted entries; the Reference/CPU apply is
     *  order-robust. Empty when there are no processor faces. Kept last so existing positional
     *  aggregate initialisations of CommunicationPattern remain valid.
     *
     */
    std::vector<localIdx> offDiagRowSortPerm;
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

/**
 * @brief Returns a cached CommunicationPattern for the given mesh partition.
 *
 * Computes the pattern via `computeCommunicationPattern` on first call and
 * memoises the result in `mesh.stencilDB()`. Subsequent calls for the same
 * mesh return a reference to the cached value without any MPI collectives.
 * This amortises the `MPI_Alltoall` / `MPI_Alltoallv` cost over many halo
 * exchanges on an immutable mesh topology.
 *
 * @param mesh  The local mesh partition. Must remain alive for the duration
 *              of any exchange that holds the returned reference.
 * @return      A const reference to the memoised CommunicationPattern stored
 *              in mesh.stencilDB(). Valid for the mesh lifetime.
 */
const CommunicationPattern& cachedCommunicationPattern(const UnstructuredMesh& mesh);

} // namespace NeoN

#endif // NF_WITH_MPI_SUPPORT
