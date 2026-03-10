// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once


#include <vector>

#include "NeoN/core/primitives/label.hpp"

namespace NeoN
{

struct CommunicationPattern
{

    // indices which values in bMatrix need to be communicated
    std::vector<localIdx> commIdx;

    // number of elements to send neighbouring ranks
    std::vector<int> sendCounts;

    mpi::Environment env;
};


}
