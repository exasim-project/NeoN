// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>

#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/halfDuplexCommBuffer.hpp"
#include "NeoN/core/view.hpp"

namespace NeoN
{

#ifdef NF_WITH_MPI_SUPPORT

namespace mpi
{

/**
 * @class FullDuplexCommBuffer
 * @brief A buffer for full-duplex communication in a distributed system using MPI.
 *
 * The FullDuplexCommBuffer class facilitates efficient, non-blocking, point-to-point data
 * exchange between MPI ranks, allowing for simultaneous send and receive operations. It
 * manages two HalfDuplexCommBuffer instances: one for sending data and one for receiving data.
 */
class FullDuplexCommBuffer
{
public:

    /**
     * @brief Default constructor.
     */
    FullDuplexCommBuffer() = default;

    /**
     * @brief Constructor that initializes the send and receive buffers.
     * @param environ The MPI environment.
     * @param sendSize The number of nodes, per rank, that this rank sends to.
     * @param receiveSize The number of nodes, per rank, that this rank receives from.
     */
    FullDuplexCommBuffer(
        MPIEnvironment mpiEnviron,
        std::vector<std::size_t> sendSize,
        std::vector<std::size_t> receiveSize
    )
        : send_(mpiEnviron, sendSize), receive_(mpiEnviron, receiveSize) {};

    /**
     * @brief Check if the communication buffers are initialized.
     * @return True if the buffers are initialized, false otherwise.
     */
    inline bool isCommInit() const { return send_.isCommInit() && receive_.isCommInit(); }

    /**
     * @brief Initialize the communication buffer.
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param commName A name for the communication, typically a file and line number.
     */
    template<typename valueType>
    void initComm(std::string commName)
    {
        send_.initComm<valueType>(commName);
        receive_.initComm<valueType>(commName);
    }

    /**
     * @brief Gets a View of data for the send buffer for a specific rank.
     * @tparam valueType The type of the data.
     * @param rank The rank of the send buffer to get.
     * @return A view of data for the send buffer.
     */
    template<typename valueType>
    View<valueType> getSend(const size_t rank)
    {
        return send_.get<valueType>(rank);
    }

    /**
     * @brief Gets a view of data for the send buffer for a specific rank.
     * @tparam valueType The type of the data.
     * @param rank The rank of the send buffer to get.
     * @return A view of data for the send buffer.
     */
    template<typename valueType>
    View<const valueType> getSend(const size_t rank) const
    {
        return send_.get<valueType>(rank);
    }

    /**
     * @brief Gets a view of data for the receive buffer for a specific rank.
     * @tparam valueType The type of the data.
     * @param rank The rank of the receive buffer to get.
     * @return A view of data for the receive buffer.
     */
    template<typename valueType>
    View<valueType> getReceive(const size_t rank)
    {
        return receive_.get<valueType>(rank);
    }

    /**
     * @brief Gets a view of data for the receive buffer for a specific rank.
     * @tparam valueType The type of the data.
     * @param rank The rank of the receive buffer to get.
     * @return A view of data for the receive buffer.
     */
    template<typename valueType>
    View<const valueType> getReceive(const size_t rank) const
    {
        return receive_.get<valueType>(rank);
    }

    /**
     * @brief Start non-blocking communication by sending and receiving data.
     */
    inline void startComm()
    {
        send_.send();
        receive_.receive();
    }

    /**
     * @brief Check if the communication is complete.
     * @return True if the communication is complete, false otherwise.
     */
    inline bool isComplete() { return send_.isComplete() && receive_.isComplete(); }

    /**
     * @brief Blocking wait for the communication to complete.
     */
    inline void waitComplete()
    {
        send_.waitComplete();
        receive_.waitComplete();
    }

    /**
     * @brief Finalize the communication by cleaning up the buffers.
     */
    inline void finaliseComm()
    {
        send_.finaliseComm();
        receive_.finaliseComm();
    }

private:

    HalfDuplexCommBuffer send_;    /**< The send buffer. */
    HalfDuplexCommBuffer receive_; /**< The receive buffer. */
};

} // namespace mpi

#endif

}
