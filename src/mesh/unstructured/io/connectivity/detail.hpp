// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/core/vector/vector.hpp"

#include <vector>

namespace NeoN::io::detail
{

// Build a SegmentedVector from flat values and their per-segment sizes (on exec).
inline SegmentedVector<localIdx, localIdx> makeSegmentedVector(
    const std::vector<localIdx>& flatValues,
    const std::vector<localIdx>& sizes,
    const Executor& exec
)
{
    SerialExecutor serial;

    std::vector<localIdx> offsets;
    offsets.reserve(sizes.size() + 1);
    offsets.push_back(0);
    for (localIdx sz : sizes)
    {
        offsets.push_back(offsets.back() + sz);
    }

    return SegmentedVector<localIdx, localIdx>(
        Vector<localIdx>(serial, flatValues).copyToExecutor(exec),
        Vector<localIdx>(serial, offsets).copyToExecutor(exec)
    );
}

} // namespace NeoN::io::detail
