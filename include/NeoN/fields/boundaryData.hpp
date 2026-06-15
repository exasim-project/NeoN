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
    void communicate(std::pair<localIdx, localIdx> range, int neighborRank)
    {
        const auto [rangeStart, rangeEnd] = range;
        const localIdx patchSize = rangeEnd - rangeStart;

        mpi::Environment mpiEnv;

        // Find or insert a persistent pool entry for this patch (keyed by rangeStart).
        // Proc-patch count is typically 1-4 per rank; a linear scan is effectively O(1).
        auto it = std::find_if(
            pool_.begin(),
            pool_.end(),
            [rangeStart](const CommBuffer& b) { return b.rangeStart == rangeStart; }
        );
        if (it == pool_.end())
        {
            pool_.push_back(CommBuffer {});
            it = pool_.end() - 1;
            it->rangeStart = rangeStart;
        }
        CommBuffer& buf = *it;
        buf.patchSize = patchSize;

        const auto byteCount =
            static_cast<mpi_label_t>(patchSize) * static_cast<mpi_label_t>(sizeof(ValueType));
        const auto neighborRankLabel = static_cast<mpi_label_t>(neighborRank);

        // Deterministic, symmetric tag for the processor patch shared by (myRank, neighborRank).
        // A unique tag per unordered rank pair, identical on both sides, makes each isend/irecv
        // match its true partner regardless of posting order. min*P+max is symmetric so both
        // ranks of the pair compute the same tag.
        const auto nProcs = static_cast<mpi_label_t>(mpiEnv.sizeRank());
        const auto myRankLabel = static_cast<mpi_label_t>(mpiEnv.rank());
        const mpi_label_t pairKey = std::min(myRankLabel, neighborRankLabel) * nProcs
                                  + std::max(myRankLabel, neighborRankLabel);
        const mpi_label_t tagUb = static_cast<mpi_label_t>(mpiEnv.tagUpperBound());
        const mpi_label_t pairTag = pairKey % tagUb;
        NF_ASSERT(
            pairTag < tagUb,
            "pairTag " << pairTag << " >= MPI_TAG_UB " << tagUb << "; nProcs=" << nProcs
        );

        const bool useGpuPath = mpiEnv.gpuAwareMpi() && std::holds_alternative<GPUExecutor>(exec_);

        MPI_Request sendReq, recvReq;
        if (useGpuPath)
        {
            // Grow-only: only (re)allocate when the buffer is absent or too small.
            if (!buf.deviceRecvBuf
                || buf.deviceRecvBuf->size() < static_cast<std::size_t>(patchSize))
                buf.deviceRecvBuf = Vector<ValueType>(exec_, patchSize, ValueType {});
            mpi::isend<char>(
                reinterpret_cast<const char*>(value_.data() + rangeStart),
                byteCount,
                neighborRankLabel,
                pairTag,
                mpiEnv.comm(),
                &sendReq
            );
            mpi::irecv<char>(
                reinterpret_cast<char*>(buf.deviceRecvBuf->data()),
                byteCount,
                neighborRankLabel,
                pairTag,
                mpiEnv.comm(),
                &recvReq
            );
        }
        else
        {
            // Grow-only resize: only extend when the current capacity is insufficient.
            if (static_cast<localIdx>(buf.sendBuf.size()) < patchSize)
            {
                buf.sendBuf.resize(static_cast<std::size_t>(patchSize));
                buf.recvBuf.resize(static_cast<std::size_t>(patchSize));
            }
            NF_DEBUG_ASSERT(
                static_cast<localIdx>(buf.sendBuf.size()) >= patchSize,
                "sendBuf capacity " << buf.sendBuf.size() << " < patchSize " << patchSize
            );
            // Stage exactly patchSize elements from the patch range (device or CPU -> host).
            // Both arguments to std::visit must be Executor variants, not bare alternatives.
            std::visit(
                detail::deepCopyVisitor<ValueType>(
                    patchSize, value_.data() + rangeStart, buf.sendBuf.data()
                ),
                exec_,                      // source executor (device or CPU)
                Executor(SerialExecutor {}) // dest executor (host)
            );
            mpi::isend<char>(
                reinterpret_cast<const char*>(buf.sendBuf.data()),
                byteCount,
                neighborRankLabel,
                pairTag,
                mpiEnv.comm(),
                &sendReq
            );
            mpi::irecv<char>(
                reinterpret_cast<char*>(buf.recvBuf.data()),
                byteCount,
                neighborRankLabel,
                pairTag,
                mpiEnv.comm(),
                &recvReq
            );
        }
        communicating_ = true;
        requests_.push_back(sendReq);
        requests_.push_back(recvReq);
        activeKeys_.push_back(rangeStart);
    }

    // Retained as a latent diagnostic helper (not used in the drain path after
    // waitAll() was updated to call mpi::waitAll). isComplete() is kept to preserve
    // symmetry with HalfDuplexCommBuffer::isComplete() and for potential future use.
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

    // Test-only observability of the host send-staging buffer for a given patch key.
    // Returns the buffer data() pointer so a test can assert pointer identity (no realloc)
    // across communicate()/waitAll() rounds. Returns nullptr / 0 when no entry exists.
    const ValueType* sendBufPtrForTest(localIdx rangeStart) const
    {
        for (const auto& b : pool_)
            if (b.rangeStart == rangeStart) return b.sendBuf.data();
        return nullptr;
    }
    std::size_t sendBufCapForTest(localIdx rangeStart) const
    {
        for (const auto& b : pool_)
            if (b.rangeStart == rangeStart) return b.sendBuf.capacity();
        return 0;
    }
    std::size_t sendBufSizeForTest(localIdx rangeStart) const
    {
        for (const auto& b : pool_)
            if (b.rangeStart == rangeStart) return b.sendBuf.size();
        return 0;
    }
    std::size_t poolSizeForTest() const { return pool_.size(); }

#endif

    void waitAll() const
    {
#ifdef NF_WITH_MPI_SUPPORT
        if (requests_.empty() || !communicating_) return;
        mpi::waitAll(requests_);
        mpi::Environment mpiEnv;
        const bool useGpuPath = mpiEnv.gpuAwareMpi() && std::holds_alternative<GPUExecutor>(exec_);
        if (useGpuPath)
        {
            // Iterate only the patches posted this round (not the full pool).
            for (const localIdx key : activeKeys_)
            {
                CommBuffer& buf = *std::find_if(
                    pool_.begin(),
                    pool_.end(),
                    [key](const CommBuffer& b) { return b.rangeStart == key; }
                );
                auto srcView = buf.deviceRecvBuf->view();
                auto dstView = value_.view();
                const localIdx start = buf.rangeStart;
                parallelFor(
                    exec_,
                    {0, buf.patchSize},
                    KOKKOS_LAMBDA(const localIdx k) { dstView[start + k] = srcView[k]; }
                );
            }
            // LOAD-BEARING FENCE: the parallelFor copy-back above (deviceRecvBuf -> value_) is
            // asynchronous on the GPUExecutor. deviceRecvBuf is now a persistent pool member
            // reused across rounds; fencing here guarantees the device kernel has completed before
            // the next communicate() posts MPI_Irecv into the same deviceRecvBuf allocation
            // (use-before-reuse). Removing this fence is a data race on device memory.
            fence(exec_);
        }
        else
        {
            // Iterate only the patches posted this round; copy back exactly patchSize elements.
            // Both arguments to std::visit must be Executor variants, not bare alternatives.
            for (const localIdx key : activeKeys_)
            {
                CommBuffer& buf = *std::find_if(
                    pool_.begin(),
                    pool_.end(),
                    [key](const CommBuffer& b) { return b.rangeStart == key; }
                );
                std::visit(
                    detail::deepCopyVisitor<ValueType>(
                        buf.patchSize, buf.recvBuf.data(), value_.data() + buf.rangeStart
                    ),
                    Executor(SerialExecutor {}), // source executor (host)
                    Executor(exec_)              // dest executor (device or CPU)
                );
            }
        }
        requests_.clear();
        communicating_ = false;
        activeKeys_.clear(); // retains capacity; pool_ persists for reuse next round
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
        std::vector<ValueType> sendBuf;                 // host staging: lazy, grow-only
        std::vector<ValueType> recvBuf;                 // host staging: lazy, grow-only
        std::optional<Vector<ValueType>> deviceRecvBuf; // device buffer: lazy, grow-only
        localIdx rangeStart {-1};                       // pool key
        localIdx patchSize {0};                         // current capacity watermark
    };
    mutable std::vector<MPI_Request>
        requests_; ///< Per-round MPI request handles (send+recv pairs). clear() retains capacity.
    mutable std::vector<CommBuffer>
        pool_; ///< Persistent staging-buffer pool keyed by rangeStart. Never cleared after warm-up.
    mutable std::vector<localIdx>
        activeKeys_; ///< Per-round list of rangeStarts posted this round. clear() retains capacity.
    mutable bool communicating_ = false;
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
