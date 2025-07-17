// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/communicator.hpp"

namespace NeoN
{

bool Communicator::isComplete(std::string commName)
{
    NF_DEBUG_ASSERT(
        CommBuffer_.find(commName) != CommBuffer_.end(),
        "No communication buffer associated with key: " << commName
    );
    return CommBuffer_[commName]->isComplete();
}

Communicator::bufferType* Communicator::findDuplexBuffer()
{
    for (auto it = buffers.begin(); it != buffers.end(); ++it)
        if (!it->isCommInit()) return &(*it);
    return nullptr;
}

Communicator::bufferType* Communicator::createNewDuplexBuffer()
{
    // determine buffer size.
    std::vector<std::size_t> rankSendSize(mpiEnviron_.sizeRank());
    std::vector<std::size_t> rankReceiveSize(mpiEnviron_.sizeRank());
    for (size_t rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
    {
        rankSendSize[rank] = sendMap_[rank].size();
        rankReceiveSize[rank] = receiveMap_[rank].size();
    }
    buffers.emplace_back(mpi::FullDuplexCommBuffer(mpiEnviron_, rankSendSize, rankReceiveSize));
    return &buffers.back();
}
};
