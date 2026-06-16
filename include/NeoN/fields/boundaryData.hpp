// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"

#include <vector>
#include <utility>
#include <cstdio>
#include <algorithm>

#ifdef NF_WITH_MPI_SUPPORT
#include <mpi.h>
#include <optional>
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#endif

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
template<typename ValueType>
class BoundaryData
{

public:

    using BoundaryDataType = ValueType;

    ~BoundaryData()
    {
        // The staging pool backs the memory of any in-flight MPI_Isend/MPI_Irecv calls.
        // Destroying pool_ while operations are pending is undefined behaviour, so
        // drain all outstanding requests before the storage is freed.
        waitAll();
    }

    /**
     * @brief Copy constructor.
     * @param rhs The boundaryVectors object to be copied.
     */
    BoundaryData(const BoundaryData<ValueType>& rhs)
        : exec_(rhs.exec_), value_(rhs.value_), refValue_(rhs.refValue_),
          valueFraction_(rhs.valueFraction_), refGrad_(rhs.refGrad_),
          boundaryTypes_(rhs.boundaryTypes_), offset_(rhs.offset_), nBoundaries_(rhs.nBoundaries_),
          nBoundaryFaces_(rhs.nBoundaryFaces_)
    {}


    /**
     * @brief Copy constructor.
     * @param rhs The boundaryVectors object to be copied.
     */
    BoundaryData(const Executor& exec, const BoundaryData<ValueType>& rhs)
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
        : exec_(exec), value_(exec, nBoundaryFaces, ValueType {}),
          refValue_(exec, nBoundaryFaces, ValueType {}),
          valueFraction_(exec, nBoundaryFaces, scalar(0)),
          refGrad_(exec, nBoundaryFaces, ValueType {}), boundaryTypes_(exec, nBoundaryTypes),
          offset_(SerialExecutor {}, nBoundaryTypes + 1), nBoundaries_(nBoundaryTypes),
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
    const Vector<ValueType>& value() const
    {
        waitAll();
        return value_;
    }

    /**
     * @brief Get the view storing the computed values from the boundary
     * condition.
     * @note calls waitAll to ensure all boundary data is updated.
     * @return The view storing the computed values.
     */
    Vector<ValueType>& value()
    {
        waitAll();
        return value_;
    }

    /** @copydoc BoundaryData::refValue()*/
    const Vector<ValueType>& refValue() const { return refValue_; }

    /**
     * @brief Get the view storing the Dirichlet boundary values.
     * @return The view storing the Dirichlet boundary values.
     */
    Vector<ValueType>& refValue() { return refValue_; }

    /** @copydoc BoundaryData::valueFraction()*/
    const Vector<scalar>& valueFraction() const { return valueFraction_; }

    /**
     * @brief Get the view storing the fraction of the boundary value.
     * @return The view storing the fraction of the boundary value.
     */
    Vector<scalar>& valueFraction() { return valueFraction_; }

    /** @copydoc BoundaryData::refGrad()*/
    const Vector<ValueType>& refGrad() const { return refGrad_; }

    /**
     * @brief Get the view storing the Neumann boundary values.
     * @return The view storing the Neumann boundary values.
     */
    Vector<ValueType>& refGrad() { return refGrad_; }

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

    BoundaryData<ValueType>& operator=(const BoundaryData<ValueType>& rhs)
    {
#ifdef NF_WITH_MPI_SUPPORT
        NF_ASSERT(
            !communicating_,
            "BoundaryData: assignment while a halo exchange is in flight is undefined. "
            "Call waitAll() before reassigning."
        );
        // The staging pool is a transient cache, not field state. Reset it so the next
        // communicate() rebuilds with correct keys and sizes (grow-only) for the assigned mesh.
        pool_.clear();
        activeKeys_.clear();
        requests_.clear();
        communicating_ = false;
#endif
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

    BoundaryData<ValueType>& operator=(const BoundaryData<ValueType>&& rhs)
    {
#ifdef NF_WITH_MPI_SUPPORT
        NF_ASSERT(
            !communicating_,
            "BoundaryData: assignment while a halo exchange is in flight is undefined. "
            "Call waitAll() before reassigning."
        );
        // The staging pool is a transient cache, not field state. Reset it so the next
        // communicate() rebuilds with correct keys and sizes (grow-only) for the assigned mesh.
        pool_.clear();
        activeKeys_.clear();
        requests_.clear();
        communicating_ = false;
#endif
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

#ifdef NF_WITH_MPI_SUPPORT
    /**
     * @brief Stage a proc-patch's owner values for deferred halo exchange.
     *
     * Records the patch range and neighbour rank for this round without posting
     * any MPI operations. The actual isend/irecv is deferred to waitAll(), which
     * is called from value() outside the correctBoundaryConditions loop. This
     * "post all, drain once" discipline prevents the eager-drain bug: if each
     * patch's recv were drained immediately, the second proc patch's in-flight
     * recv would be cancelled before completion, leaving its ghost equal to the
     * owner seed instead of the neighbour value.
     *
     * Re-keyed to per-neighbour rank (D-04): multiple proc patches to the same
     * neighbour share one CommBuffer. The unified gather+post+scatter runs in
     * waitAll() once all patches have been staged.
     *
     * @param range          [rangeStart, rangeEnd) index range in value_ for this patch.
     * @param neighborRank   The MPI rank of the neighbour that owns the far side of this patch.
     * @param pattern        CommunicationPattern (sendCounts, boundaryMapVector, env).
     *                       Must remain valid until the matching waitAll() completes.
     *                       Callers obtain this via cachedCommunicationPattern(mesh).
     * @param procFaceStart  First proc-boundary index in value_: equals
     *                       mesh.nBoundaryFaces() (physical-boundary count).
     *
     * @note Option B (extracting post-all to VolumeField::correctBoundaryConditions with a
     *       start/finish split) is the Phase 14 OVERLAP-01 migration path — not implemented here.
     */
    void communicate(
        std::pair<localIdx, localIdx> range,
        int neighborRank,
        const CommunicationPattern& pattern,
        localIdx procFaceStart
    )
    {
        // Find or insert a persistent pool entry keyed by neighbourRank.
        // Proc-neighbour count is typically 1-4 per rank; a linear scan is O(1) in practice.
        auto it = std::find_if(
            pool_.begin(),
            pool_.end(),
            [neighborRank](const CommBuffer& b) { return b.neighbourRank == neighborRank; }
        );
        if (it == pool_.end())
        {
            pool_.push_back(CommBuffer {});
            it = pool_.end() - 1;
            it->neighbourRank = neighborRank;
        }
        CommBuffer& buf = *it;

        // Grow-only: size the per-neighbour buffers to sendCounts[neighbourRank].
        // Multiple patches to the same neighbour reuse the same CommBuffer; the buffer is
        // already sized on the first communicate() call for this neighbour this round.
        const int neiCount = pattern.sendCounts[static_cast<std::size_t>(neighborRank)];
        if (static_cast<localIdx>(buf.sendBuf.size()) < static_cast<localIdx>(neiCount))
        {
            buf.sendBuf.resize(static_cast<std::size_t>(neiCount));
            buf.recvBuf.resize(static_cast<std::size_t>(neiCount));
        }
        buf.totalFaces = static_cast<localIdx>(neiCount);

        // Record this neighbour as active for this round (deduplicated: only add once).
        const bool alreadyActive =
            std::find(activeKeys_.begin(), activeKeys_.end(), neighborRank) != activeKeys_.end();
        if (!alreadyActive) activeKeys_.push_back(neighborRank);

        // Cache the pattern pointer and procFaceStart for use in waitAll().
        // The pattern lives in mesh.stencilDB() (cachedCommunicationPattern) and is valid
        // for the mesh lifetime; storing a raw pointer is safe.
        cachedPattern_ = &pattern;
        procFaceStart_ = procFaceStart;

        communicating_ = true;
        (void)range; // range is used implicitly: value_ is already seeded by
                     // updateProcBoundaryOwnerValue
    }

    // Retained as a latent diagnostic helper (not used in the drain path).
    // isComplete() can be used to poll whether all outstanding MPI operations have
    // finished without blocking (useful for debugging and diagnostics).
    bool isComplete() const
    {
        if (requests_.empty() || !communicating_) return true;
        for (auto& req : requests_)
        {
            if (!mpi::test(&req)) return false;
        }
        communicating_ = false;
        return true;
    }

    // Test-only observability of the host send-staging buffer for a given neighbour rank.
    // Returns the buffer data() pointer so a test can assert pointer identity (no realloc)
    // across communicate()/waitAll() rounds. Returns nullptr / 0 when no entry exists.
    const ValueType* sendBufPtrForTest(int neighbourRank) const
    {
        for (const auto& b : pool_)
            if (b.neighbourRank == neighbourRank) return b.sendBuf.data();
        return nullptr;
    }
    std::size_t sendBufCapForTest(int neighbourRank) const
    {
        for (const auto& b : pool_)
            if (b.neighbourRank == neighbourRank) return b.sendBuf.capacity();
        return 0;
    }
    std::size_t sendBufSizeForTest(int neighbourRank) const
    {
        for (const auto& b : pool_)
            if (b.neighbourRank == neighbourRank) return b.sendBuf.size();
        return 0;
    }
    std::size_t poolSizeForTest() const { return pool_.size(); }

#endif

    void waitAll() const
    {
#ifdef NF_WITH_MPI_SUPPORT
        if (!communicating_ || activeKeys_.empty() || cachedPattern_ == nullptr) return;

        const auto& pattern = *cachedPattern_;
        const auto& bmv = pattern.boundaryMapVector;
        const auto& sendCounts = pattern.sendCounts;
        const auto nRanks = static_cast<std::size_t>(pattern.env.sizeRank());

        // Build per-rank send displacements (prefix sum of sendCounts).
        // Processor-patch send/recv counts are symmetric, so sdispl[r] serves as both
        // the send and recv displacement for rank r.
        std::vector<int> sdispl(nRanks, 0);
        for (std::size_t r = 1; r < nRanks; ++r)
            sdispl[r] = sdispl[r - 1] + sendCounts[r - 1];

        mpi::Environment mpiEnv = pattern.env;
        const bool useGpuPath = mpiEnv.gpuAwareMpi() && std::holds_alternative<GPUExecutor>(exec_);
        const auto nProcs = static_cast<mpi_label_t>(mpiEnv.sizeRank());
        const auto myRankLabel = static_cast<mpi_label_t>(mpiEnv.rank());

        // ---- Stage: gather proc-face values from value_ into per-neighbour send buffers ----
        // For each active neighbour: gather the neiCount faces using boundaryMapVector.
        // boundaryMapVector[rankGroupedPos] = procFacePos (0-based within proc-boundary block).
        // value_[procFaceStart_ + bmv[sdispl[nei] + j]] is the proc-face to send at rank-grouped
        // position (sdispl[nei] + j) for neighbour nei.
        // Stage device -> host so we have a host buffer to hand to MPI.
        const int totalSend = sendCounts[nRanks]; // last element = total proc faces
        std::vector<ValueType> sendHostBuf(static_cast<std::size_t>(totalSend));
        {
            // Copy ALL proc-face values from device to host in one visit.
            std::visit(
                detail::deepCopyVisitor<ValueType>(
                    static_cast<localIdx>(totalSend),
                    value_.data() + procFaceStart_,
                    sendHostBuf.data()
                ),
                exec_,                      // source: device or CPU
                Executor(SerialExecutor {}) // dest: host
            );
        }

        // ---- Post per-neighbour non-blocking isend/irecv ----
        // One isend + one irecv per unique active neighbour (O(neighbours), not O(patches)).
        for (const int neiRank : activeKeys_)
        {
            const auto nei = static_cast<std::size_t>(neiRank);
            const int neiCount = sendCounts[nei];
            if (neiCount == 0) continue;

            CommBuffer& buf = *std::find_if(
                pool_.begin(),
                pool_.end(),
                [neiRank](const CommBuffer& b) { return b.neighbourRank == neiRank; }
            );

            // Gather: repack rank-grouped send buffer from proc-face-ordered host values.
            // sendHostBuf is proc-face ordered; sendBuf[sdispl[nei]+j] =
            // sendHostBuf[bmv[sdispl[nei]+j]]
            for (int j = 0; j < neiCount; ++j)
            {
                const auto rgPos = static_cast<std::size_t>(sdispl[nei] + j);
                const auto pfPos = static_cast<std::size_t>(bmv[rgPos]);
                NF_DEBUG_ASSERT(
                    static_cast<localIdx>(pfPos) < static_cast<localIdx>(totalSend),
                    "bmv[" << rgPos << "]=" << pfPos << " out of range [0," << totalSend << ")"
                );
                buf.sendBuf[static_cast<std::size_t>(j)] = sendHostBuf[pfPos];
            }

            const auto neiLabel = static_cast<mpi_label_t>(neiRank);
            const mpi_label_t pairKey =
                std::min(myRankLabel, neiLabel) * nProcs + std::max(myRankLabel, neiLabel);
            const mpi_label_t tagUb = static_cast<mpi_label_t>(mpiEnv.tagUpperBound());
            const mpi_label_t pairTag = pairKey % tagUb;
            NF_ASSERT(
                pairTag < tagUb,
                "pairTag " << pairTag << " >= MPI_TAG_UB " << tagUb << "; nProcs=" << nProcs
            );
            const auto byteCount =
                static_cast<mpi_label_t>(neiCount) * static_cast<mpi_label_t>(sizeof(ValueType));

            MPI_Request sendReq, recvReq;
            mpi::isend<char>(
                reinterpret_cast<const char*>(buf.sendBuf.data()),
                byteCount,
                neiLabel,
                pairTag,
                mpiEnv.comm(),
                &sendReq
            );
            mpi::irecv<char>(
                reinterpret_cast<char*>(buf.recvBuf.data()),
                byteCount,
                neiLabel,
                pairTag,
                mpiEnv.comm(),
                &recvReq
            );
            requests_.push_back(sendReq);
            requests_.push_back(recvReq);
        }

        // ---- Single drain (post all, drain once) ----
        // All neighbours are posted above; drain them together. Per-neighbour drain would
        // reproduce the eager-drain bug (dominant historical failure).
        mpi::waitAll(requests_);

        // ---- Scatter rank-grouped recv buffer into value_'s proc-boundary tail ----
        // value_[procFaceStart_ + boundaryMapVector[k]] = recvBuf[k]  for k in [0, totalSend)
        // Collect the full rank-grouped recv output into a single host buffer first.
        std::vector<ValueType> recvHostBuf(static_cast<std::size_t>(totalSend));
        for (const int neiRank : activeKeys_)
        {
            const auto nei = static_cast<std::size_t>(neiRank);
            const int neiCount = sendCounts[nei];
            if (neiCount == 0) continue;
            CommBuffer& buf = *std::find_if(
                pool_.begin(),
                pool_.end(),
                [neiRank](const CommBuffer& b) { return b.neighbourRank == neiRank; }
            );
            for (int j = 0; j < neiCount; ++j)
                recvHostBuf[static_cast<std::size_t>(sdispl[nei] + j)] =
                    buf.recvBuf[static_cast<std::size_t>(j)];
        }
        // Scatter: recvHostBuf is rank-grouped; bmv maps rank-grouped pos -> proc-face pos.
        std::vector<ValueType> scatterBuf(static_cast<std::size_t>(totalSend));
        for (int k = 0; k < totalSend; ++k)
        {
            const auto pfPos = static_cast<std::size_t>(bmv[static_cast<std::size_t>(k)]);
            NF_DEBUG_ASSERT(
                static_cast<int>(pfPos) < totalSend,
                "bmv[" << k << "]=" << pfPos << " out of range [0," << totalSend << ")"
            );
            scatterBuf[pfPos] = recvHostBuf[static_cast<std::size_t>(k)];
        }
        // Copy scattered result to value_[procFaceStart_..] (device or CPU).
        std::visit(
            detail::deepCopyVisitor<ValueType>(
                static_cast<localIdx>(totalSend), scatterBuf.data(), value_.data() + procFaceStart_
            ),
            Executor(SerialExecutor {}), // source: host
            Executor(exec_)              // dest: device or CPU
        );

        if (useGpuPath)
        {
            // LOAD-BEARING FENCE (COMM-03 recv): the deepCopyVisitor above launches an
            // asynchronous device kernel (device copy-back). Fencing here ensures the kernel
            // has completed before the next communicate() stages new values into value_
            // (use-before-write on device memory). Removing this fence is a data race.
            fence(exec_);
        }

        requests_.clear();
        communicating_ = false;
        activeKeys_.clear(); // retains capacity; pool_ persists for reuse next round
        cachedPattern_ = nullptr;
#endif
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

    /// Grants NoWaitAccess (and only it) access to valueNoWait().
    friend struct NoWaitAccess;

    /**
     * @brief Non-draining access to the value storage.
     * @warning Does NOT call waitAll(), so any in-flight processor-halo exchange is left pending.
     * Used by the processor boundary condition to SEED the owner value before posting its
     * isend/irecv: seeding must NOT drain a previous patch's exchange, because completing and
     * clearing the comm buffers patch-by-patch (mid correctBoundaryConditions loop) serialises the
     * halo exchange and, on a rank with two proc patches, leaves the second patch's recv
     * unmatched — the halo then silently keeps its owner seed. All patches must post first; the
     * single waitAll() triggered by the next real value() read then completes them together.
     *
     * Deliberately private: bypassing waitAll() is easy to misuse, so access goes through the
     * NoWaitAccess passkey struct, keeping the set of callers auditable.
     */
    Vector<ValueType>& valueNoWait() { return value_; }

    Executor exec_;                   ///< The executor on which the field is stored
    mutable Vector<ValueType> value_; ///< The Vector storing the computed values from the
                                      ///< boundary condition.
    Vector<ValueType> refValue_;      ///< The Vector storing the Dirichlet boundary values.
    Vector<scalar>
        valueFraction_; ///< Fraction between Dirichlet (1.0) and Neuman (0.0) boundary value
    Vector<ValueType> refGrad_; ///< The Vector storing the Neumann boundary values.
    Vector<int> boundaryTypes_; ///< The Vector storing the boundary types.
    Vector<localIdx> offset_;   ///< The Vector storing the offsets of each boundary.
    localIdx nBoundaries_;      ///< The number of boundaries.
    localIdx nBoundaryFaces_;   ///< The number of boundary faces.

#ifdef NF_WITH_MPI_SUPPORT
    struct CommBuffer
    {
        std::vector<ValueType> sendBuf; // host staging: lazy, grow-only (size = sendCounts[nei])
        std::vector<ValueType> recvBuf; // host staging: lazy, grow-only (size = sendCounts[nei])
        std::optional<Vector<ValueType>> deviceRecvBuf; // device buffer: reserved for Phase 13
        int neighbourRank {-1};  // pool key (re-keyed from rangeStart to neighbour rank)
        localIdx totalFaces {0}; // capacity watermark = sendCounts[neighbourRank]
    };
    mutable std::vector<MPI_Request>
        requests_; ///< Per-round MPI request handles (send+recv pairs). clear() retains capacity.
    mutable std::vector<CommBuffer> pool_; ///< Persistent staging-buffer pool keyed by
                                           ///< neighbourRank. Never cleared after warm-up.
    mutable std::vector<int> activeKeys_;  ///< Per-round list of neighbourRanks posted this round.
                                           ///< clear() retains capacity.
    mutable bool communicating_ = false;
    mutable const CommunicationPattern* cachedPattern_ =
        nullptr; ///< Pattern from last communicate(); valid until waitAll().
    mutable localIdx procFaceStart_ =
        0; ///< First proc-boundary index in value_ = mesh.nBoundaryFaces().
#endif
};

/**
 * @brief Passkey granting non-draining access to BoundaryData's value storage.
 *
 * BoundaryData::valueNoWait() is private because skipping waitAll() leaves in-flight
 * processor-halo exchanges pending and is easy to misuse. Callers that legitimately need it
 * (the processor boundary condition seeding owner values before posting its exchange) go
 * through this struct, so every bypass site is greppable via NoWaitAccess.
 */
struct NoWaitAccess
{
    template<typename ValueType>
    static Vector<ValueType>& value(BoundaryData<ValueType>& in)
    {
        return in.valueNoWait();
    }
};

}
