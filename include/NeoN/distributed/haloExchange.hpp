// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef NF_WITH_MPI_SUPPORT

#include <mpi.h>
#include <algorithm>
#include <vector>
#include <variant>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN
{

/**
 * @brief Unified non-blocking value-halo exchange driven by a CommunicationPattern.
 *
 * Posts one mpi::isend<char> and one mpi::irecv<char> per unique neighbour rank
 * (O(neighbours), not O(patches)), drains them all in a single wait, then
 * scatters the rank-grouped recv buffer into proc-face order via
 * pattern.boundaryMapVector.
 *
 * @tparam T  Element type (e.g. scalar, Vec3).  Sent as raw bytes (char reinterpret);
 *            mpi::getType<T>() is never called.
 *
 * @param exec       Executor that owns sendData / recvData (device or CPU).
 * @param mesh       Local mesh partition; provides neighbour-rank enumeration.
 * @param sendData   Proc-face-ordered send buffer [0, nProcBoundaryFaces); device or host.
 * @param recvData   Proc-face-ordered output buffer [0, nProcBoundaryFaces); scatter applied.
 * @param pattern    Pre-built CommunicationPattern (sendCounts, boundaryMapVector, env).
 *
 * @note GPU send path: fence(exec) is called BEFORE staging from device so that any
 *       kernel writing sendData has completed before MPI reads it (COMM-03, load-bearing).
 * @note GPU recv path: fence(exec) is called AFTER the device copy-back to prevent reuse
 *       of host staging buffers before the copy kernel completes (COMM-03, load-bearing).
 */
template<typename T>
void haloExchange(
    const Executor& exec,
    const UnstructuredMesh& mesh,
    const T* sendData,
    T* recvData,
    const CommunicationPattern& pattern
)
{
    if (pattern.boundaryMapVector.empty()) return;

    const auto totalRecv = static_cast<int>(pattern.boundaryMapVector.size());

    mpi::Environment mpiEnv = pattern.env;

    const auto nRanks = static_cast<std::size_t>(mpiEnv.sizeRank());

    // Per-rank send displacement = prefix sum of pattern.sendCounts. sendCounts is sized nRanks+1;
    // entries [0,nRanks) are per-rank face counts, entry [nRanks] holds the total.
    //
    // Across a processor patch the shared-face count is symmetric (every proc face is shared by
    // exactly two ranks), so this rank's recv count from peer r equals its send count to peer r.
    // The recv block therefore has the same size and displacement as the send block, and
    // boundaryMapVector — built on the recv side from that same displacement — addresses the recv
    // buffer directly. No allToAll is needed here: the run-invariant comm metadata already lives in
    // the pattern, so the primitive itself stays collective-free.
    const auto& sendCountsVec = pattern.sendCounts;
    std::vector<int> sdispl(nRanks, 0);
    for (std::size_t r = 1; r < nRanks; ++r)
        sdispl[r] = sdispl[r - 1] + sendCountsVec[r - 1];
    const int totalSend = sendCountsVec[nRanks]; // last element = total

    // Determine whether GPU-direct path is available.
    // gpuAwareMpi() defaults to true; set NEON_FORCE_HOST_BUFFER to force host staging.
    const bool useGpuPath = mpiEnv.gpuAwareMpi() && std::holds_alternative<GPUExecutor>(exec);

    // ----- Stage sendData from device (or CPU) to host send buffer -----
    // sendBuf is rank-grouped: sendBuf[k] = sendData[boundaryMapVector[k]],
    // because boundaryMapVector[rankGroupedPos] = procFacePos.
    std::vector<T> procFaceHostSend(static_cast<std::size_t>(totalSend));
    if (useGpuPath)
    {
        // COMM-03 send fence: flush any device kernel that is still writing sendData
        // before MPI reads the device pointer (or before we stage from device).
        fence(exec); // COMM-03 send fence — load-bearing; do not remove
    }
    // Stage from device (or CPU) to host staging buffer.
    std::visit(
        detail::deepCopyVisitor<T>(
            static_cast<localIdx>(totalSend), sendData, procFaceHostSend.data()
        ),
        exec,                       // source: device or CPU
        Executor(SerialExecutor {}) // dest: host
    );

    // Repack proc-face ordered host buffer into rank-grouped send buffer.
    std::vector<T> sendBuf(static_cast<std::size_t>(totalSend));
    for (int k = 0; k < totalRecv; ++k)
    {
        const auto procFaceIdx =
            static_cast<std::size_t>(pattern.boundaryMapVector[static_cast<std::size_t>(k)]);
        sendBuf[static_cast<std::size_t>(k)] = procFaceHostSend[procFaceIdx];
    }

    // Recv buffer: rank-grouped layout, size = totalRecv.
    std::vector<T> recvBuf(static_cast<std::size_t>(totalRecv));

    // ----- Post per-neighbour non-blocking send/recv -----
    // One isend + one irecv per unique neighbour rank (O(neighbours), not O(patches)).
    // Deterministic, symmetric pair-tag: min(myRank,neiRank)*nProcs + max(myRank,neiRank),
    // identical on both ends of the pair — ensures correct matching regardless of post order.
    const auto nProcs = static_cast<mpi_label_t>(mpiEnv.sizeRank());
    const auto myRankLabel = static_cast<mpi_label_t>(mpiEnv.rank());

    std::vector<MPI_Request> requests;
    requests.reserve(2 * nRanks);

    for (std::size_t ni = 0; ni < nRanks; ++ni)
    {
        if (sendCountsVec[ni] == 0) continue;

        const auto neighborRankLabel = static_cast<mpi_label_t>(ni);
        const mpi_label_t pairKey = std::min(myRankLabel, neighborRankLabel) * nProcs
                                  + std::max(myRankLabel, neighborRankLabel);
        const mpi_label_t tagUb = static_cast<mpi_label_t>(mpiEnv.tagUpperBound());
        const mpi_label_t pairTag = pairKey % tagUb;
        NF_ASSERT(
            pairTag < tagUb,
            "pairTag " << pairTag << " >= MPI_TAG_UB " << tagUb << "; nProcs=" << nProcs
        );

        if (sendCountsVec[ni] > 0)
        {
            const auto byteCount =
                static_cast<mpi_label_t>(sendCountsVec[ni]) * static_cast<mpi_label_t>(sizeof(T));
            MPI_Request sendReq;
            mpi::isend<char>(
                reinterpret_cast<const char*>(sendBuf.data() + sdispl[ni]),
                byteCount,
                neighborRankLabel,
                pairTag,
                mpiEnv.comm(),
                &sendReq
            );
            requests.push_back(sendReq);
        }

        if (sendCountsVec[ni] > 0)
        {
            const auto byteCount =
                static_cast<mpi_label_t>(sendCountsVec[ni]) * static_cast<mpi_label_t>(sizeof(T));
            MPI_Request recvReq;
            mpi::irecv<char>(
                reinterpret_cast<char*>(recvBuf.data() + sdispl[ni]),
                byteCount,
                neighborRankLabel,
                pairTag,
                mpiEnv.comm(),
                &recvReq
            );
            requests.push_back(recvReq);
        }
    }

    // ----- Single drain -----
    // Post ALL neighbours first, THEN drain once.  Per-neighbour drain reproduces
    // the eager-drain bug (dominant historical failure, proc-halo-mispair root cause).
    mpi::waitAll(requests);

    // ----- Scatter rank-grouped recv buffer into proc-face ordered recvData -----
    // recvData[boundaryMapVector[k]] = recvBuf[k]  for k in [0, totalRecv)
    // (procFaceStart = 0 here since recvData is a standalone proc-face buffer;
    //  callers targeting BoundaryData::value_ set procFaceStart = mesh.nBoundaryFaces()).
    NF_DEBUG_ASSERT(
        static_cast<int>(pattern.boundaryMapVector.size()) == totalRecv,
        "boundaryMapVector size " << pattern.boundaryMapVector.size() << " != totalRecv "
                                  << totalRecv
    );

    // Scatter on host into a host-side output buffer, then copy back to device if needed.
    std::vector<T> recvHostScattered(static_cast<std::size_t>(totalRecv));
    for (int k = 0; k < totalRecv; ++k)
    {
        const auto procFaceIdx =
            static_cast<std::size_t>(pattern.boundaryMapVector[static_cast<std::size_t>(k)]);
        NF_DEBUG_ASSERT(
            static_cast<int>(procFaceIdx) < totalRecv,
            "boundaryMapVector[" << k << "] = " << procFaceIdx << " out of range [0, " << totalRecv
                                 << ")"
        );
        recvHostScattered[procFaceIdx] = recvBuf[static_cast<std::size_t>(k)];
    }

    // Copy scattered host output to recvData (device or CPU).
    std::visit(
        detail::deepCopyVisitor<T>(
            static_cast<localIdx>(totalRecv), recvHostScattered.data(), recvData
        ),
        Executor(SerialExecutor {}), // source: host
        exec                         // dest: device or CPU
    );

    if (useGpuPath)
    {
        // COMM-03 recv fence: ensure the device copy-back kernel has completed before
        // the caller reuses recvData on the device (prevents use-before-write on GPU).
        fence(exec); // COMM-03 recv fence — load-bearing; do not remove
    }
}

} // namespace NeoN

#endif // NF_WITH_MPI_SUPPORT
