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

#include <vector>
#include <utility>

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
        // commBuffers_ backs the memory of any in-flight MPI_Isend/MPI_Irecv calls.
        // Destroying them while operations are pending is undefined behaviour, so
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
        CommBuffer buf;
        buf.rangeStart = rangeStart;
        buf.patchSize = patchSize;

        const auto byteCount =
            static_cast<mpi_label_t>(patchSize) * static_cast<mpi_label_t>(sizeof(ValueType));
        const auto neighborRankLabel = static_cast<mpi_label_t>(neighborRank);

        const bool useGpuPath = mpiEnv.gpuAwareMpi() && std::holds_alternative<GPUExecutor>(exec_);

        MPI_Request sendReq, recvReq;
        if (useGpuPath)
        {
            buf.deviceRecvBuf = Vector<ValueType>(exec_, patchSize, ValueType {});
            mpi::isend<char>(
                reinterpret_cast<const char*>(value_.data() + rangeStart),
                byteCount,
                neighborRankLabel,
                0,
                mpiEnv.comm(),
                &sendReq
            );
            mpi::irecv<char>(
                reinterpret_cast<char*>(buf.deviceRecvBuf->data()),
                byteCount,
                neighborRankLabel,
                0,
                mpiEnv.comm(),
                &recvReq
            );
        }
        else
        {
            auto valH = value_.copyToHost();
            buf.sendBuf.resize(static_cast<std::size_t>(patchSize));
            buf.recvBuf.resize(static_cast<std::size_t>(patchSize));
            for (localIdx k = 0; k < patchSize; k++)
                buf.sendBuf[static_cast<std::size_t>(k)] = valH.view()[rangeStart + k];
            mpi::isend<char>(
                reinterpret_cast<const char*>(buf.sendBuf.data()),
                byteCount,
                neighborRankLabel,
                0,
                mpiEnv.comm(),
                &sendReq
            );
            mpi::irecv<char>(
                reinterpret_cast<char*>(buf.recvBuf.data()),
                byteCount,
                neighborRankLabel,
                0,
                mpiEnv.comm(),
                &recvReq
            );
        }
        communicating_ = true;
        requests_.push_back(sendReq);
        requests_.push_back(recvReq);
        commBuffers_.push_back(std::move(buf));
    }

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


#endif

    void waitAll() const
    {
#ifdef NF_WITH_MPI_SUPPORT
        if (requests_.empty() || !communicating_) return;
        while (!isComplete())
        {
        }
        mpi::Environment mpiEnv;
        const bool useGpuPath = mpiEnv.gpuAwareMpi() && std::holds_alternative<GPUExecutor>(exec_);
        if (useGpuPath)
        {
            for (const auto& buf : commBuffers_)
            {
                auto srcView = buf.deviceRecvBuf->view();
                auto dstView = value_.view();
                const localIdx start = buf.rangeStart;
                parallelFor(
                    exec_,
                    {0, buf.patchSize},
                    KOKKOS_LAMBDA(const localIdx k) { dstView[start + k] = srcView[k]; }
                );
            }
        }
        else
        {
            auto valH = value_.copyToHost();
            for (const auto& buf : commBuffers_)
            {
                for (localIdx k = 0; k < buf.patchSize; k++)
                    valH.view()[buf.rangeStart + k] = buf.recvBuf[static_cast<std::size_t>(k)];
            }
            value_ = valH.copyToExecutor(exec_);
        }
        requests_.clear();
        communicating_ = false;
        commBuffers_.clear();
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
        std::vector<ValueType> sendBuf;                 // host staging, used when !gpuAwareMpi
        std::vector<ValueType> recvBuf;                 // host staging, used when !gpuAwareMpi
        std::optional<Vector<ValueType>> deviceRecvBuf; // device buffer, used when gpuAwareMpi
        localIdx rangeStart;
        localIdx patchSize;
    };
    mutable std::vector<MPI_Request>
        requests_; ///< Pending MPI requests (send+recv pairs per patch).
    mutable std::vector<CommBuffer>
        commBuffers_; ///< Send/recv staging buffers for pending requests.
    mutable bool communicating_ = false;
#endif
};

/**@brief exchange values on processor boundaries*/
template<typename ValueType>
void communicateBoundaryData(
    const CommunicationPattern& commPattern,
    const std::vector<localIdx> procPatchOffset,
    Vector<ValueType>& boundaryData
)
{
    // // 2. exchange
    auto mpiEnv = commPattern.env;
    auto commRanks = mpiEnv.sizeRank();
    auto sendSize = commPattern.sendCounts[commRanks];

    // compute send displacements
    auto sdispls = std::vector<int>(commRanks, 0);
    int j = 0;
    for (int i = 0; i < sdispls.size(); i++)
    {
        // check if rank communicates data
        if (commPattern.sendCounts[i] > 0)
        {
            // send displ patch start
            sdispls[i] = procPatchOffset[j];
            j++;
        }
    }

    // FIXME recvBuffer does not need to same size as boundaryData
    auto recvBuffer = Vector<ValueType>(boundaryData.exec(), boundaryData.size());
    auto copyIdx = Vector<localIdx>(boundaryData.exec(), procPatchOffset);
    MPI_Alltoallv(
        boundaryData.data(),
        commPattern.sendCounts.data(),
        sdispls.data(),
        mpi::getType<ValueType>(),
        recvBuffer.data(),
        commPattern.sendCounts.data(),
        sdispls.data(),
        mpi::getType<ValueType>(),
        mpiEnv.comm()
    );

    // FIXME TODO this might not be needed communication happens directly into boundaryData
    auto exec = boundaryData.exec();
    auto outV = boundaryData.view();
    const auto idxV = copyIdx.view();
    const auto inV = recvBuffer.view();

    parallelFor(
        exec,
        {0, idxV.size()},
        NEON_LAMBDA(const localIdx i) { outV[idxV[i]] = inV[idxV[i]]; },
        "copyMap"
    );
}
}
