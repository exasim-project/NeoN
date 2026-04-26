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

/**@brief exchange values on processor boundaries*/
template<typename ValueType>
void communicateBoundaryData(
    const CommunicationPattern& commPattern,
    const std::vector<std::pair<localIdx, localIdx>> procPatchOffset,
    Vector<ValueType>& boundaryData
)
{
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
            auto [start, end] = procPatchOffset[j];
            sdispls[i] = start;
            j++;
        }
    }

    // FIXME recvBuffer does not need to same size as boundaryData
    auto recvBuffer = Vector<ValueType>(boundaryData.exec(), boundaryData.size());
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
    // auto copyIdx = Vector<localIdx>(boundaryData.exec(), procPatchOffset);
    auto exec = boundaryData.exec();
    auto outV = boundaryData.view();
    // const auto idxV = copyIdx.view();
    const auto inV = recvBuffer.view();

    parallelFor(
        exec,
        {0, procPatchOffset.size()},
        NEON_LAMBDA(const localIdx p) {
            auto [start, end] = procPatchOffset[p];
            for (auto i = start; i < end; i++)
            {
                outV[i] = inV[i];
            }
        },
        "copyMap"
    );
}

// Specialization for Vec3
template<>
inline void communicateBoundaryData(
    const CommunicationPattern& commPattern,
    const std::vector<std::pair<localIdx, localIdx>> procPatchOffset,
    Vector<Vec3>& boundaryData
)
{
    auto mpiEnv = commPattern.env;
    auto commRanks = mpiEnv.sizeRank();
    auto sendSize = commPattern.sendCounts[commRanks];

    // compute send displacements
    std::vector<localIdx> sendCounts(commPattern.sendCounts.size(), 0);
    for (int i = 0; i < sendCounts.size(); i++)
    {
        sendCounts[i] = commPattern.sendCounts[i] * 3;
    }

    auto sdispls = std::vector<int>(commRanks, 0);
    int j = 0;
    for (int i = 0; i < sdispls.size(); i++)
    {
        // check if rank communicates data
        if (sendCounts[i] > 0)
        {
            // send displ patch start
            auto [start, end] = procPatchOffset[j];
            sdispls[i] = 3 * start;
            j++;
        }
    }

    // FIXME recvBuffer does not need to be same size as boundaryData
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

    auto recvBuffer = Vector<NeoN::scalar>(boundaryData.exec(), 3 * boundaryData.size());
    auto recvBufferV = recvBuffer.view();
    MPI_Alltoallv(
        sendBuffer.data(),
        sendCounts.data(),
        sdispls.data(),
        mpi::getType<scalar>(),
        recvBuffer.data(),
        sendCounts.data(),
        sdispls.data(),
        mpi::getType<scalar>(),
        mpiEnv.comm()
    );

    const auto inV = recvBuffer.view();
    auto outV = boundaryData.view();
    parallelFor(
        exec,
        {0, procPatchOffset.size()},
        NEON_LAMBDA(const localIdx p) {
            auto [start, end] = procPatchOffset[p];
            for (auto i = start; i < end; i++)
            {
                outV[i][0] = inV[3 * i + 0];
                outV[i][1] = inV[3 * i + 1];
                outV[i][2] = inV[3 * i + 2];
            }
        },
        "copyMap"
    );
}

}
