// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"

#include <vector>

namespace NeoN::test
{

/// Build a CellConnectivity with NeoN types from host vectors.
inline io::CellConnectivity makeCellConn(
    const Executor& exec, std::vector<std::vector<localIdx>> cells, std::vector<int32_t> types
)
{
    SerialExecutor serial;
    std::vector<localIdx> values, offsets;
    offsets.push_back(0);
    for (auto& c : cells)
    {
        values.insert(values.end(), c.begin(), c.end());
        offsets.push_back(static_cast<localIdx>(offsets.back() + static_cast<localIdx>(c.size())));
    }
    return io::CellConnectivity {
        SegmentedVector<localIdx, localIdx>(
            Vector<localIdx>(serial, values).copyToExecutor(exec),
            Vector<localIdx>(serial, offsets).copyToExecutor(exec)
        ),
        Vector<int32_t>(serial, types).copyToExecutor(exec),
        static_cast<localIdx>(cells.size())
    };
}

} // namespace NeoN::test
