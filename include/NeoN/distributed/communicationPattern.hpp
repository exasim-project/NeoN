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
     *  TODO: currently dead — computeCommunicationPattern always constructs this empty and no code
     *  reads it. Either drop it, or populate and consume it. To adopt it: fill it so that
     *  boundaryMapVector[k] is the local processor-boundary face index of the k-th received halo
     *  entry (in recvIdx order). That mapping is what a *value*-based halo exchange driven by this
     *  pattern needs to scatter received neighbour data into a field's proc tail — which would let
     *  the remaining per-field exchanges (the geometry-scheme neighbour-centre exchange and the
     *  field-halo path BoundaryData::communicate) share this single CommunicationPattern instead of
     *  each rolling its own MPI isend/irecv.
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

} // namespace NeoN

#endif // NF_WITH_MPI_SUPPORT
