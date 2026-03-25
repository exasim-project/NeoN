// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once


#include <vector>

#include "NeoN/core/primitives/label.hpp"

namespace NeoN
{

/** @struct a struct collecting all required data for distributed communication
 */
struct CommunicationPattern
{

    // indices which values in bMatrix need to be communicated
    std::vector<localIdx> commIdx;

    // number of elements to send neighbouring ranks
    // where sendCounts[comm.size] = total number of send elements
    std::vector<int> sendCounts;

    std::vector<localIdx> boundaryMapVector;

    mpi::Environment env;
};


}
