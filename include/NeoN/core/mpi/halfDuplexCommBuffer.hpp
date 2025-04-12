// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <span>
#include <string>
#include <typeindex>
#include <vector>

#include "NeoN/core/error.hpp"
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"

namespace NeoN
{

#ifdef NF_WITH_MPI_SUPPORT

namespace mpi
{

/**
 * @brief A hash function for a string.
 *
 * @param str The string to be hashed.
 * @return int The hash value of the string.
 */
inline int bufferHash(const std::string& str)
{
    // There is also an MPI environment value for that, but somehow it doesn't work using that
    // Also reserve 10 tags for other uses
    constexpr int maxTagValue = 32767 - 10;
    std::size_t tag = std::hash<std::string> {}(str);
    tag &= 0x7FFFFFFF; // turn 'int' signed bit to 0, MPI does not like negative tags.
    return (static_cast<int>(tag) % maxTagValue) + 10;
}

/**
 * @class HalfDuplexCommBuffer
 * @brief A data buffer for half-duplex communication in a distributed system using MPI.
 *
 * The HalfDuplexCommBuffer class facilitates efficient, non-blocking, point-to-point data exchange
 * between MPI ranks in a distributed system. It maintains a buffer used for data communication,
 * capable of handling various data types. The buffer does not shrink once initialized, minimizing
 * memory reallocation and improving memory efficiency. The class operates in a half-duplex mode,
 * meaning it is either sending or receiving data at any given time.
 */
class HalfDuplexCommBuffer
{

public:

    /**
     * @brief Default constructor
     */
    HalfDuplexCommBuffer() = default;

    /**
     * @brief Default destructor
     */
    ~HalfDuplexCommBuffer() = default;

    /**
     * @brief Construct a new Half Duplex Buffer object
     *
     * @param mpiEnviron The MPI environment.
     * @param rankCommSize The number of nodes per rank to be communicated with.
     */
    HalfDuplexCommBuffer(MPIEnvironment mpiEnviron, std::vector<size_t> rankCommSize)
        : mpiEnviron_(mpiEnviron)
    {
        setCommRankSize<char>(rankCommSize);
    }

    /**
     * @brief Set the MPI environment.
     *
     * @param mpiEnviron The MPI environment.
     */
    inline void setMPIEnvironment(MPIEnvironment mpiEnviron) { mpiEnviron_ = mpiEnviron; }

    /**
     * @brief Get the communication name.
     *
     * @return const std::string& The name of the communication.
     */
    inline bool isCommInit() const { return tag_ != -1; }

    /**
     * @brief Set the buffer data size based on the rank communication and value type.
     *
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param rankCommSize The number of nodes per rank to be communicated with.
     */
    template<typename valueType>
    void setCommRankSize(std::vector<size_t> rankCommSize)
    {
        NF_DEBUG_ASSERT(
            !isCommInit(), "Communication buffer was initialised by name: " << commName_ << "."
        );
        NF_DEBUG_ASSERT(
            rankCommSize.size() == mpiEnviron_.sizeRank(),
            "Rank size mismatch. " << rankCommSize.size() << " vs. " << mpiEnviron_.sizeRank()
        );
        typeSize_ = sizeof(valueType);
        rankOffset_.resize(rankCommSize.size() + 1);
        request_.resize(rankCommSize.size(), MPI_REQUEST_NULL);
        updateDataSize([&](const size_t rank) { return rankCommSize[rank]; }, sizeof(valueType));
    }

    /**
     * @brief Initialize the communication buffer.
     *
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param commName A name for the communication, typically a file and line number.
     */
    template<typename valueType>
    void initComm(std::string commName)
    {
        NF_DEBUG_ASSERT(
            !isCommInit(), "Communication buffer was initialised by name: " << commName_ << "."
        );
        setType<valueType>();
        commName_ = commName;
        tag_ = bufferHash(commName);
    }

    /**
     * @brief Get the communication name.
     *
     * @return const std::string& The name of the communication.
     */
    inline const std::string& getCommName() const { return commName_; }

    /**
     * @brief Check if the communication is complete.
     * @return true if the communication is complete else false.
     */
    bool isComplete();

    /**
     * @brief Post send for data to begin sending to all ranks this rank communicates with.
     */
    void send();

    /**
     * @brief Post receive for data to begin receiving from all ranks this rank communicates
     * with.
     */
    void receive();

    /**
     * @brief Blocking wait for the communication to finish.
     */
    void waitComplete();

    /**
     * @brief Finalise the communication.
     */
    void finaliseComm();

    /**
     * @brief Get a span of the buffer data for a given rank.
     *
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param rank The rank of the data to be retrieved.
     * @return std::span<valueType> A span of the data for the given rank.
     */
    template<typename valueType>
    std::span<valueType> get(const size_t rank)
    {
        NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
        NF_DEBUG_ASSERT(typeSize_ == sizeof(valueType), "Data type (size) mismatch.");
        return std::span<valueType>(
            reinterpret_cast<valueType*>(rankBuffer_.data() + rankOffset_[rank]),
            (rankOffset_[rank + 1] - rankOffset_[rank]) / sizeof(valueType)
        );
    }

    /**
     * @brief Get a span of the buffer data for a given rank.
     *
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param rank The rank of the data to be retrieved.
     * @return std::span<const valueType> A span of the data for the given rank.
     */
    template<typename valueType>
    std::span<const valueType> get(const size_t rank) const
    {
        NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
        NF_DEBUG_ASSERT(typeSize_ == sizeof(valueType), "Data type (size) mismatch.");
        return std::span<const valueType>(
            reinterpret_cast<const valueType*>(rankBuffer_.data() + rankOffset_[rank]),
            (rankOffset_[rank + 1] - rankOffset_[rank]) / sizeof(valueType)
        );
    }

private:

    int tag_ {-1};                        /*< The tag for the communication. */
    std::string commName_ {"unassigned"}; /*< The name of the communication. */
    std::size_t typeSize_ {sizeof(char)}; /*< The data type currently stored in the buffer. */
    MPIEnvironment mpiEnviron_;           /*< The MPI environment. */
    std::vector<MPI_Request> request_;    /*< The MPI request for communication with each rank. */
    std::vector<char> rankBuffer_;        /*< The buffer data for all ranks. Never shrinks. */
    std::vector<std::size_t>
        rankOffset_; /*< The offset (in bytes) for a rank data in the buffer. */

    /**
     * @brief Set the data type for the buffer.
     */
    template<typename valueType>
    void setType()
    {
        NF_DEBUG_ASSERT(
            !isCommInit(), "Communication buffer was initialised by name: " << commName_ << "."
        );
        if (0 == (typeSize_ - sizeof(valueType))) return;
        updateDataSize(
            [rankOffset = rankOffset_, typeSize = typeSize_](const size_t rank)
            { return (rankOffset[rank + 1] - rankOffset[rank]) / typeSize; },
            sizeof(valueType)
        );

        typeSize_ = sizeof(valueType);
    }

    /**
     * @brief Update the buffer data size using a lambda (for the size per rank) and type size.
     *
     * @tparam func The lambda type for the rank size.
     * @param rankSize The function to get the rank size.
     * @param dataSize The size of the data to be stored in the buffer.
     */
    template<typename func>
    void updateDataSize(func rankSize, std::size_t newSize)
    {
        std::size_t dataSize = 0;
        for (size_t rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
        {
            rankOffset_[rank] = dataSize;
            dataSize += rankSize(rank) * newSize;
        }
        rankOffset_.back() = dataSize;
        if (rankBuffer_.size() < dataSize) rankBuffer_.resize(dataSize); // we never size down.
    }
};

} // namespace mpi

#endif

}
