// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <tuple>

#include "NeoN/core/macros.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/fields/boundaryData.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void SurfaceField<ValueType>::correctBoundaryConditions()
{
    // Identify processor patches by patchID. Processor patches are the trailing
    // patches in the boundary mesh (same convention used in
    // basicGeometryScheme::collectProcPatchOffsets). SurfaceBoundary does not
    // expose name()/getName() like VolumeBoundary does, so we use the patchID
    // range instead.
    const auto& bm = this->mesh().boundaryMesh();
    const auto& nbrRanks = bm.neighbourRank();
    const auto totalPatches = bm.nBoundaries();
    const auto procPatchCount = bm.nProcBoundaryPatches();
    const auto firstProcPatch = totalPatches - procPatchCount;

    // Collect (neighbourRank, start, end) and sort by ascending neighbourRank.
    // communicateBoundaryData expects procPatchOffset in ascending neighbour-rank
    // order; mesh-order is decomposition-dependent and may not match.
    std::vector<std::tuple<localIdx, localIdx, localIdx>> procTriples;
    for (auto& boundaryCondition : boundaryConditions_)
    {
        boundaryCondition.correctBoundaryCondition(this->field_);
        if (procPatchCount > 0 && boundaryCondition.patchID() >= firstProcPatch)
        {
            const auto procIdx = boundaryCondition.patchID() - firstProcPatch;
            auto [start, end] = boundaryCondition.range();
            procTriples.emplace_back(nbrRanks[procIdx], start, end);
        }
    }

    if (!procTriples.empty())
    {
        std::sort(
            procTriples.begin(),
            procTriples.end(),
            [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); }
        );
        std::vector<std::pair<localIdx, localIdx>> procPatchOffset;
        procPatchOffset.reserve(procTriples.size());
        for (const auto& t : procTriples)
        {
            procPatchOffset.emplace_back(std::get<1>(t), std::get<2>(t));
        }

        // FIXME dont recompute communication pattern
        auto commPattern = computeCommunicationPattern(this->mesh());
        communicateBoundaryData(commPattern, procPatchOffset, this->field_.boundaryData().value());
    }
}

#define NN_DECLARE_SURFACE_FIELD(TYPENAME) template class SurfaceField<TYPENAME>

NN_FOR_ALL_VALUE_TYPES(NN_DECLARE_SURFACE_FIELD);

}
