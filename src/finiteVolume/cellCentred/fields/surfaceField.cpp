// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/fields/boundaryData.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void SurfaceField<ValueType>::correctBoundaryConditions()
{
    auto procPatchOffset = std::vector<std::pair<localIdx, localIdx>> {};
    for (auto& boundaryCondition : boundaryConditions_)
    {
        boundaryCondition.correctBoundaryCondition(this->field_);
        if (boundaryCondition.name() == "processor")
        {
            auto [start, end] = boundaryCondition.range();
            procPatchOffset.emplace_back(start, end);
        }
    }

    if (procPatchOffset.size() > 0)
    {
        // FIXME dont recompute communication pattern
        auto commPattern = computeCommunicationPattern(this->mesh());
        communicateBoundaryData(commPattern, procPatchOffset, this->field_.boundaryData().value());
    }
}

#define NN_DECLARE_SURFACE_FIELD(TYPENAME) template class SurfaceField<TYPENAME>

NN_FOR_ALL_VALUE_TYPES(NN_DECLARE_SURFACE_FIELD);

}
