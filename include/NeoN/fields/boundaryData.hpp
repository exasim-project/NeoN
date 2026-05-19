// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/core/mpi/operators.hpp"

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <utility>

#include <mpi.h>

namespace NeoN
{


/**
 * @class BoundaryData
 * @brief Represents the boundary fields for a computational domain.
 *
 * The BoundaryData class stores the boundary conditions and related
 * information for a computational domain. It provides access to the computed
 * values, reference values, value fractions, reference gradients, boundary
 * types, offsets, and the number of boundaries and boundary faces.
 *
 * @tparam ValueType The type of the underlying field values
 */
template<typename T>
class BoundaryData
{

public:

    /**
     * @brief Copy constructor.
     * @param rhs The boundaryVectors object to be copied.
     */
    BoundaryData(const BoundaryData<T>& rhs)
        : exec_(rhs.exec_), value_(rhs.value_), refValue_(rhs.refValue_),
          valueFraction_(rhs.valueFraction_), refGrad_(rhs.refGrad_),
          boundaryTypes_(rhs.boundaryTypes_), offset_(rhs.offset_), nBoundaries_(rhs.nBoundaries_),
          nBoundaryFaces_(rhs.nBoundaryFaces_)
    {}


    /**
     * @brief Copy constructor.
     * @param rhs The boundaryVectors object to be copied.
     */
    BoundaryData(const Executor& exec, const BoundaryData<T>& rhs)
        : exec_(rhs.exec_), value_(exec, rhs.value_), refValue_(exec, rhs.refValue_),
          valueFraction_(exec, rhs.valueFraction_), refGrad_(exec, rhs.refGrad_),
          boundaryTypes_(exec, rhs.boundaryTypes_), offset_(SerialExecutor {}, rhs.offset_),
          nBoundaries_(rhs.nBoundaries_), nBoundaryFaces_(rhs.nBoundaryFaces_)
    {}


    /**
     * @brief constructor with default initialized Vectors from sizes.
     * @param exec - The executor
     * @param nBoundaryFaces - The total number of boundary faces
     * @param nBoundaryType - The total number of boundary patches
     */
    BoundaryData(const Executor& exec, localIdx nBoundaryFaces, localIdx nBoundaryTypes)
        : exec_(exec), value_(exec, nBoundaryFaces, T {}), refValue_(exec, nBoundaryFaces, T {}),
          valueFraction_(exec, nBoundaryFaces, scalar(0)), refGrad_(exec, nBoundaryFaces, T {}),
          boundaryTypes_(exec, nBoundaryTypes, int(0)),
          offset_(SerialExecutor {}, nBoundaryTypes + 1, localIdx(0)), nBoundaries_(nBoundaryTypes),
          nBoundaryFaces_(nBoundaryFaces)
    {}

    /**
     * @brief constructor from a given offsets vector
     * @warn all members except offsets are default constructed
     * @param exec - The executor
     * @param offsets - The total number of boundary faces
     */
    BoundaryData(const Executor& exec, const std::vector<localIdx>& offsets)
        : BoundaryData(exec, offsets.back(), static_cast<localIdx>(offsets.size() - 1))
    {
        offset_ = Vector(SerialExecutor {}, offsets);
    }


    /** @copydoc BoundaryData::value()*/
    const Vector<T>& value() const { return value_; }

    /**
     * @brief Get the view storing the computed values from the boundary
     * condition.
     * @return The view storing the computed values.
     */
    Vector<T>& value() { return value_; }

    /** @copydoc BoundaryData::refValue()*/
    const Vector<T>& refValue() const { return refValue_; }

    /**
     * @brief Get the view storing the Dirichlet boundary values.
     * @return The view storing the Dirichlet boundary values.
     */
    Vector<T>& refValue() { return refValue_; }

    /** @copydoc BoundaryData::valueFraction()*/
    const Vector<scalar>& valueFraction() const { return valueFraction_; }

    /**
     * @brief Get the view storing the fraction of the boundary value.
     * @return The view storing the fraction of the boundary value.
     */
    Vector<scalar>& valueFraction() { return valueFraction_; }

    /** @copydoc BoundaryData::refGrad()*/
    const Vector<T>& refGrad() const { return refGrad_; }

    /**
     * @brief Get the view storing the Neumann boundary values.
     * @return The view storing the Neumann boundary values.
     */
    Vector<T>& refGrad() { return refGrad_; }

    /**
     * @brief Get the view storing the boundary types.
     * @return The view storing the boundary types.
     */
    const Vector<int>& boundaryTypes() const { return boundaryTypes_; }

    /**
     * @brief Get the view storing the offsets of each boundary.
     * @return The view storing the offsets of each boundary.
     */
    const Vector<localIdx>& offset() const { return offset_; }

    /**
     * @brief Get the number of boundaries.
     * @return The number of boundaries.
     */
    localIdx nBoundaries() const { return nBoundaries_; }

    /**
     * @brief Get the number of boundary faces.
     * @return The number of boundary faces.
     */
    localIdx nBoundaryFaces() const { return nBoundaryFaces_; }

    /**
     * @brief Get the number of boundary faces for this patch.
     * @return The number of boundary faces for this patch.
     */
    localIdx nBoundaryFaces(localIdx patchId) const
    {
        return offset_.data()[patchId + 1] - offset_.data()[patchId];
    }

    const Executor& exec() { return exec_; }

    BoundaryData<T>& operator=(const BoundaryData<T>& rhs)
    {

        // TODO maybe dont overwrite nBoundaries and nBoundaryFaces
        // but use them for a sanity check
        nBoundaries_ = rhs.nBoundaries_;
        nBoundaryFaces_ = rhs.nBoundaryFaces_;

        value_ = rhs.value_;
        refValue_ = rhs.refValue_;
        valueFraction_ = rhs.valueFraction_;
        refGrad_ = rhs.refGrad_;
        boundaryTypes_ = rhs.boundaryTypes_;
        offset_ = rhs.offset_;
        return *this;
    }

    BoundaryData<T>& operator=(const BoundaryData<T>&& rhs)
    {

        // TODO maybe dont overwrite nBoundaries and nBoundaryFaces
        // but use them for a sanity check
        nBoundaries_ = rhs.nBoundaries_;
        nBoundaryFaces_ = rhs.nBoundaryFaces_;

        value_ = std::move(rhs.value_);
        refValue_ = std::move(rhs.refValue_);
        valueFraction_ = std::move(rhs.valueFraction_);
        refGrad_ = std::move(rhs.refGrad_);
        boundaryTypes_ = std::move(rhs.boundaryTypes_);
        offset_ = std::move(rhs.offset_);
        return *this;
    }

    /**
     * @brief Get the range for a given patchId
     * @return The number of boundary faces.
     */
    std::pair<localIdx, localIdx> range(localIdx patchId) const
    {
        return {offset_.data()[patchId], offset_.data()[patchId + 1]};
    }

private:

    Executor exec_;                ///< The executor on which the field is stored
    Vector<T> value_;              ///< The Vector storing the computed values from the
                                   ///< boundary condition.
    Vector<T> refValue_;           ///< The Vector storing the Dirichlet boundary values.
    Vector<scalar> valueFraction_; ///< The Vector storing the fraction of
                                   ///< the boundary value.
    Vector<T> refGrad_;            ///< The Vector storing the Neumann boundary values.
    Vector<int> boundaryTypes_;    ///< The Vector storing the boundary types.
    Vector<localIdx> offset_;      ///< The Vector storing the offsets of each boundary.
    localIdx nBoundaries_;         ///< The number of boundaries.
    localIdx nBoundaryFaces_;      ///< The number of boundary faces.
};

/**@brief exchange values on processor boundaries
 *
 * @param procPatchOffset (start, end) ranges of each proc patch in the
 *        boundaryData layout, IN MESH-BOUNDARY ORDER.
 * @param targetRanks for each entry of procPatchOffset, the neighbour rank it
 *        communicates with. Must be parallel to procPatchOffset.
 *
 * Mesh-order is decomposition-dependent — patch 0 may target rank 7 and patch
 * 1 may target rank 3. The MPI Alltoallv displacement for rank r MUST be the
 * start offset of the mesh-order patch that targets r, NOT a running sum over
 * ranks. The previous implementation assumed mesh-order matched ascending-
 * rank order and shipped wrong data to wrong ranks otherwise.
 *
 * Mesh-order is also what downstream consumers (e.g. setProcBoundarySparsity-
 * Pattern) expect for the resulting layout — sorting procPatchOffset by rank
 * inside this routine would corrupt the matrix sparsity row/col pairing.
 */
template<typename ValueType>
void communicateBoundaryData(
    const CommunicationPattern& commPattern,
    const std::vector<std::pair<localIdx, localIdx>> procPatchOffset,
    const std::vector<int>& targetRanks,
    Vector<ValueType>& boundaryData
)
{
    auto mpiEnv = commPattern.env;
    auto commRanks = mpiEnv.sizeRank();
    auto sendSize = commPattern.sendCounts[commRanks];

    // For each mesh-order patch, point sdispls[targetRank] at that patch's
    // start offset in boundaryData. Ranks not in targetRanks keep the default
    // 0 displacement (sendCounts[r] == 0 means MPI reads no bytes from there).
    NF_ASSERT(
        procPatchOffset.size() == targetRanks.size(),
        "procPatchOffset and targetRanks must have the same length"
    );
    auto sdispls = std::vector<int>(commRanks, 0);
    for (std::size_t p = 0; p < procPatchOffset.size(); ++p)
    {
        sdispls[targetRanks[p]] = static_cast<int>(procPatchOffset[p].first);
    }

    // MPI-01 fix: derive per-rank recv counts via MPI_Alltoall on send counts.
    auto recvCountsVec = std::vector<int>(commRanks, 0);
    {
        auto sendCountsInt = std::vector<int>(commRanks, 0);
        for (int r = 0; r < commRanks; ++r)
            sendCountsInt[r] = static_cast<int>(commPattern.sendCounts[r]);
        MPI_Alltoall(
            sendCountsInt.data(), 1, MPI_INT, recvCountsVec.data(), 1, MPI_INT, mpiEnv.comm()
        );
    }
    auto rdispls = std::vector<int>(commRanks, 0);
    for (int r = 1; r < commRanks; ++r)
        rdispls[r] = rdispls[r - 1] + recvCountsVec[r - 1];
    int totalRecv = (commRanks > 0) ? rdispls.back() + recvCountsVec.back() : 0;

    // Flush any pending GPU kernels writing to boundaryData before MPI reads it.
    // Kernels launched by per-BC correctBoundaryCondition() in the caller are async
    // on the GPU executor; without this fence MPI would send pre-kernel device data.
    deviceSync(boundaryData.exec());

#if defined(NEON_CUDA_AWARE_MPI)
    auto recvBuffer = Vector<ValueType>(boundaryData.exec(), static_cast<localIdx>(totalRecv));
    MPI_Alltoallv(
        boundaryData.data(),
        commPattern.sendCounts.data(),
        sdispls.data(),
        mpi::getType<ValueType>(),
        recvBuffer.data(),
        recvCountsVec.data(),
        rdispls.data(),
        mpi::getType<ValueType>(),
        mpiEnv.comm()
    );
#else
    // Host-stage: D→H before MPI, H→D after (safe for non-CUDA-aware MPI / WSL2 OpenMPI)
    auto sendHost = boundaryData.copyToHost(); // Vector<ValueType> on SerialExecutor
    auto recvHost = Vector<ValueType>(SerialExecutor {}, static_cast<localIdx>(totalRecv));
    const bool ccbdTrace = (std::getenv("NF_PROC_BC_TRACE") != nullptr);
    if (ccbdTrace)
    {
        int rk = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);
        // Print first 4 entries from sendHost at the proc-patch offset for
        // each target rank, so we can see what's actually being SENT.
        std::fprintf(
            stderr,
            "[NF_PROC_BC_TRACE][rank %d][communicateBoundaryData] commRanks=%d "
            "totalRecv=%d\n",
            rk,
            commRanks,
            totalRecv
        );
        for (int r = 0; r < commRanks; ++r)
        {
            std::fprintf(
                stderr,
                "[NF_PROC_BC_TRACE][rank %d][communicateBoundaryData]   "
                "to_rank=%d sendCount=%d sdispl=%d  recvCount=%d rdispl=%d\n",
                rk,
                r,
                (int)commPattern.sendCounts[r],
                sdispls[r],
                recvCountsVec[r],
                rdispls[r]
            );
        }
        const auto sV = sendHost.view();
        for (std::size_t p = 0; p < procPatchOffset.size(); ++p)
        {
            const auto s = procPatchOffset[p].first;
            const auto e = procPatchOffset[p].second;
            if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
            {
                std::fprintf(
                    stderr,
                    "[NF_PROC_BC_TRACE][rank %d][communicateBoundaryData] "
                    "send patch p=%zu to_rank=%d range=[%lld,%lld) "
                    "first4=[%.6e %.6e %.6e %.6e]\n",
                    rk,
                    p,
                    targetRanks[p],
                    (long long)s,
                    (long long)e,
                    (e - s) > 0 ? (double)sV[s + 0] : 0.0,
                    (e - s) > 1 ? (double)sV[s + 1] : 0.0,
                    (e - s) > 2 ? (double)sV[s + 2] : 0.0,
                    (e - s) > 3 ? (double)sV[s + 3] : 0.0
                );
            }
        }
    }
    MPI_Alltoallv(
        sendHost.data(),
        commPattern.sendCounts.data(),
        sdispls.data(),
        mpi::getType<ValueType>(),
        recvHost.data(),
        recvCountsVec.data(),
        rdispls.data(),
        mpi::getType<ValueType>(),
        mpiEnv.comm()
    );
    if (ccbdTrace)
    {
        int rk = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);
        const auto rV = recvHost.view();
        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
            std::fprintf(
                stderr,
                "[NF_PROC_BC_TRACE][rank %d][communicateBoundaryData] "
                "POST-MPI recvHost first8=[%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e]\n",
                rk,
                totalRecv > 0 ? (double)rV[0] : 0.0,
                totalRecv > 1 ? (double)rV[1] : 0.0,
                totalRecv > 2 ? (double)rV[2] : 0.0,
                totalRecv > 3 ? (double)rV[3] : 0.0,
                totalRecv > 4 ? (double)rV[4] : 0.0,
                totalRecv > 5 ? (double)rV[5] : 0.0,
                totalRecv > 6 ? (double)rV[6] : 0.0,
                totalRecv > 7 ? (double)rV[7] : 0.0
            );
        }
    }
    // H→D copy back; Vector(exec, hostVec) calls deepCopyVisitor host→device
    auto recvBuffer = Vector<ValueType>(boundaryData.exec(), recvHost);
#endif

    auto exec = boundaryData.exec();
    // On GPU, UCX may fill recvBuffer via a private CUDA stream that is not the
    // Kokkos stream. fence() calls cudaDeviceSynchronize() which waits for ALL
    // CUDA streams — ensuring the received data is visible before the unpack kernel.
    deviceSync(exec);
    auto outV = boundaryData.view();
    const auto inV = recvBuffer.view();

    // Build a per-patch descriptor: [outStart, outEnd, inStart] so the unpack
    // kernel knows both where to write (outStart..outEnd in boundaryData) and
    // where to read (inStart in recvBuffer, which is rdispls[targetRank]).
    // Mirror to a device-resident Vector — capturing host data in a GPU lambda
    // dereferences host memory on device and trips Kokkos bounds checks.
    auto offsetHost = std::vector<localIdx>(3 * procPatchOffset.size());
    for (std::size_t p = 0; p < procPatchOffset.size(); p++)
    {
        offsetHost[3 * p] = procPatchOffset[p].first;
        offsetHost[3 * p + 1] = procPatchOffset[p].second;
        offsetHost[3 * p + 2] = static_cast<localIdx>(rdispls[targetRanks[p]]);
    }
    auto offsetVec = Vector<localIdx>(exec, offsetHost);
    const auto offsetV = offsetVec.view();

    parallelFor(
        exec,
        {0, procPatchOffset.size()},
        NEON_LAMBDA(const localIdx p) {
            const auto outStart = offsetV[3 * p];
            const auto outEnd = offsetV[3 * p + 1];
            const auto inStart = offsetV[3 * p + 2];
            for (localIdx j = 0; j < outEnd - outStart; j++)
            {
                outV[outStart + j] = inV[inStart + j];
            }
        },
        "copyMap"
    );

    // Ensure the unpack kernel completes before callers continue to use boundaryData.
    deviceSync(exec);
}

// Specialization for Vec3
template<>
inline void communicateBoundaryData(
    const CommunicationPattern& commPattern,
    const std::vector<std::pair<localIdx, localIdx>> procPatchOffset,
    const std::vector<int>& targetRanks,
    Vector<Vec3>& boundaryData
)
{
    auto mpiEnv = commPattern.env;
    auto commRanks = mpiEnv.sizeRank();
    auto sendSize = commPattern.sendCounts[commRanks];

    // compute send displacements. MPI_Alltoallv requires `int` count/displacement
    // arrays — using std::vector<int> here matches the MPI signature regardless of
    // how localIdx is configured at build time.
    std::vector<int> sendCounts(commPattern.sendCounts.size(), 0);
    for (int i = 0; i < sendCounts.size(); i++)
    {
        sendCounts[i] = commPattern.sendCounts[i] * 3;
    }

    // MPI-01 fix: derive per-rank recv counts from the 3x-multiplied send counts.
    auto recvCountsVec = std::vector<int>(commRanks, 0);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCountsVec.data(), 1, MPI_INT, mpiEnv.comm());
    auto rdispls = std::vector<int>(commRanks, 0);
    for (int r = 1; r < commRanks; ++r)
        rdispls[r] = rdispls[r - 1] + recvCountsVec[r - 1];
    int totalRecv3 = (commRanks > 0) ? rdispls.back() + recvCountsVec.back() : 0;

    // Per-rank displacement using the mesh-order target rank for each patch.
    // See the scalar overload above for rationale.
    NF_ASSERT(
        procPatchOffset.size() == targetRanks.size(),
        "procPatchOffset and targetRanks must have the same length"
    );
    auto sdispls = std::vector<int>(commRanks, 0);
    for (std::size_t p = 0; p < procPatchOffset.size(); ++p)
    {
        sdispls[targetRanks[p]] = 3 * static_cast<int>(procPatchOffset[p].first);
    }

    auto exec = boundaryData.exec();
    auto boundaryDataSize = boundaryData.size();
    auto sendBuffer = Vector<NeoN::scalar>(boundaryData.exec(), 3 * boundaryData.size());
    auto sendBufferV = sendBuffer.view();
    auto boundaryDataV = boundaryData.view();

    parallelFor(
        exec,
        {0, boundaryData.size()},
        NEON_LAMBDA(const localIdx i) {
            sendBufferV[3 * i + 0] = boundaryDataV[i][0];
            sendBufferV[3 * i + 1] = boundaryDataV[i][1];
            sendBufferV[3 * i + 2] = boundaryDataV[i][2];
        },
        "copyMap"
    );

    // Flush the pack kernel above (and any caller-launched BC kernels) before MPI
    // reads sendBuffer. parallelFor is async on GPU.
    deviceSync(exec);

#if defined(NEON_CUDA_AWARE_MPI)
    auto recvBuffer = Vector<NeoN::scalar>(boundaryData.exec(), static_cast<localIdx>(totalRecv3));
    MPI_Alltoallv(
        sendBuffer.data(),
        sendCounts.data(),
        sdispls.data(),
        mpi::getType<scalar>(),
        recvBuffer.data(),
        recvCountsVec.data(),
        rdispls.data(),
        mpi::getType<scalar>(),
        mpiEnv.comm()
    );
#else
    // Host-stage sendBuffer (the packed scalar array from parallelFor above).
    // NOTE: stage sendBuffer NOT boundaryData — Vec3 overload packs into sendBuffer first.
    // sendBuffer is already fenced at line 398; the D→H copy is safe.
    auto sendHost = sendBuffer.copyToHost(); // Vector<NeoN::scalar> on SerialExecutor
    auto recvHost = Vector<NeoN::scalar>(SerialExecutor {}, static_cast<localIdx>(totalRecv3));
    MPI_Alltoallv(
        sendHost.data(),
        sendCounts.data(), // already 3x-multiplied — unchanged
        sdispls.data(),
        mpi::getType<scalar>(),
        recvHost.data(),
        recvCountsVec.data(),
        rdispls.data(),
        mpi::getType<scalar>(),
        mpiEnv.comm()
    );
    // H→D copy back to device before unpack parallelFor reads recvBuffer
    auto recvBuffer = Vector<NeoN::scalar>(boundaryData.exec(), recvHost);
#endif

    // On GPU, UCX may fill recvBuffer via a private CUDA stream. fence() calls
    // cudaDeviceSynchronize() to flush all CUDA streams before the unpack kernel.
    deviceSync(exec);
    const auto inV = recvBuffer.view();
    auto outV = boundaryData.view();

    // Build a per-patch descriptor: [outStart, outEnd, inStart (in scalar units)]
    // mirroring rdispls[targetRank] so the unpack kernel reads from the correct
    // position in the compacted recvBuffer. Mirror to device — see scalar overload.
    auto offsetHost = std::vector<localIdx>(3 * procPatchOffset.size());
    for (std::size_t p = 0; p < procPatchOffset.size(); p++)
    {
        offsetHost[3 * p] = procPatchOffset[p].first;
        offsetHost[3 * p + 1] = procPatchOffset[p].second;
        offsetHost[3 * p + 2] = static_cast<localIdx>(rdispls[targetRanks[p]]);
    }
    auto offsetVec = Vector<localIdx>(exec, offsetHost);
    const auto offsetV = offsetVec.view();

    parallelFor(
        exec,
        {0, procPatchOffset.size()},
        NEON_LAMBDA(const localIdx p) {
            const auto outStart = offsetV[3 * p];
            const auto outEnd = offsetV[3 * p + 1];
            const auto inStart = offsetV[3 * p + 2]; // scalar offset in recvBuffer
            for (localIdx j = 0; j < outEnd - outStart; j++)
            {
                outV[outStart + j][0] = inV[inStart + 3 * j + 0];
                outV[outStart + j][1] = inV[inStart + 3 * j + 1];
                outV[outStart + j][2] = inV[inStart + 3 * j + 2];
            }
        },
        "copyMap"
    );

    // Ensure the unpack kernel completes before callers continue to use boundaryData.
    deviceSync(exec);
}

}
