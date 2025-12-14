// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <unordered_map>
#include <memory>


#include "NeoN/fields/field.hpp"

#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/fullDuplexCommBuffer.hpp"
#include "NeoN/core/mpi/operators.hpp"
#include "NeoN/core/mpi/environment.hpp"
#endif

namespace NeoN
{

#ifdef NF_WITH_MPI_SUPPORT
/**
 * @brief Represents a pair of indices for rank local and global indexing.
 */
struct NodeCommMap
{
    label local_idx; /**< The local index. */
};

/**
 * @brief Represents a mapping of NodeCommMap for a rank.
 */
using RankCommMap = std::vector<NodeCommMap>;

/**
 * @brief Represents, for a single map, a mapping of all RankCommMaps for either send or receive.
 */
using CommMap = std::vector<RankCommMap>;

/**
 * @class Communicator
 * @brief Manages communication between ranks in a parallel environment.
 * The Communicator class provides functionality to manage communication of field data exchange
 * between MPI ranks for unstructured meshes. The class maintains an MPI environment and maps for
 * rank-specific send and receive operations.
 */
class Communicator
{
public:

    using bufferType = mpi::FullDuplexCommBuffer;

    /**
     * @brief Default constructor.
     */
    Communicator() = default;

    /**
     * @brief Destructor.
     */
    ~Communicator() = default;

    /**
     * @brief Constructor that initializes the Communicator with MPI environment, rank send map, and
     * rank receive map.
     * @param mpiEnviron The MPI environment.
     * @param rankSendMap The rank send map.
     * @param rankReceiveMap The rank receive map.
     */
    Communicator(mpi::Environment mpiEnviron, CommMap rankSendMap, CommMap rankReceiveMap)
        : mpiEnviron_(mpiEnviron), sendMap_(rankSendMap), receiveMap_(rankReceiveMap)
    {
        NF_DEBUG_ASSERT(
            mpiEnviron_.sizeRank() == rankSendMap.size(),
            "Size of rankSendSize does not match MPI size."
        );
        NF_DEBUG_ASSERT(
            mpiEnviron_.sizeRank() == rankReceiveMap.size(),
            "Size of rankReceiveSize does not match MPI size."
        );
    };

    /**
     * @brief Starts the non-blocking communication for a given field and communication name.
     * @tparam valueType The value type of the field.
     * @param field The field to be communicated/synchronized.
     * @param commName The communication name, typically a file and line number.
     */
    template<typename valueType>
    void startComm(Vector<valueType>& field, const std::string& commName)
    {
        NF_DEBUG_ASSERT(
            CommBuffer_.find(commName) == CommBuffer_.end() || (!CommBuffer_[commName]),
            "There is already an ongoing communication for key " << commName << "."
        );

        CommBuffer_[commName] = findDuplexBuffer();
        if (!CommBuffer_[commName])
        {
            CommBuffer_[commName] = createNewDuplexBuffer();
        }

        CommBuffer_[commName]->initComm<valueType>(commName);
        auto fieldV = field.view();
        for (size_t rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
        {
            auto rankBuffer = CommBuffer_[commName]->getSend<valueType>(rank);
            for (size_t data = 0; data < sendMap_[rank].size(); ++data)
                rankBuffer[data] = fieldV[static_cast<size_t>(sendMap_[rank][data].local_idx)];
        }
        CommBuffer_[commName]->startComm();
    }

    /**
     * @brief Checks if the non-blocking communication with the given name is complete.
     * @param commName The communication name, typically a file and line number.
     * @return True if the communication is complete, false otherwise.
     */
    bool isComplete(std::string commName);

    /**
     * @brief Finalizes the non-blocking communication for a given field and communication name.
     * @tparam valueType The value type of the field.
     * @param field The field to be communicated/synchronized.
     * @param commName The communication name, typically a file and line number.
     */
    template<typename valueType>
    void finaliseComm(Vector<valueType>& field, std::string commName)
    {
        NF_DEBUG_ASSERT(
            CommBuffer_.find(commName) != CommBuffer_.end() && CommBuffer_[commName],
            "No communication associated with key: " << commName
        );

        CommBuffer_[commName]->waitComplete();
        auto fieldV = field.view();
        for (size_t rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
        {
            auto rankBuffer = CommBuffer_[commName]->getReceive<valueType>(rank);
            for (size_t data = 0; data < receiveMap_[rank].size(); ++data)
                fieldV[static_cast<size_t>(receiveMap_[rank][data].local_idx)] = rankBuffer[data];
        }
        CommBuffer_[commName]->finaliseComm();
        CommBuffer_[commName] = nullptr;
    }

private:

    mpi::Environment mpiEnviron_;    /**< The MPI environment. */
    CommMap sendMap_;                /**< The rank send map. */
    CommMap receiveMap_;             /**< The rank receive map. */
    std::vector<bufferType> buffers; /**< Communication buffers. */
    std::unordered_map<std::string, bufferType*>
        CommBuffer_; /**< The communication key to buffer map, nullptr indicates no assigned buffer.
                      */

    /**
     * @brief Finds an uninitialized communication buffer.
     * @return An pointer to a free communication buffer, or nullptr if no free buffer is found.
     */
    bufferType* findDuplexBuffer();

    /**
     * @brief Creates a new communication buffer with the given name.
     * @return pointer to the newly created buffer.
     */
    bufferType* createNewDuplexBuffer();
};
#endif

} // namespace NeoN
