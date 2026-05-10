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

    // Collect proc-patch (start, end) ranges in MESH-BOUNDARY ORDER, paired
    // with their target ranks. communicateBoundaryData uses targetRanks for
    // per-rank Alltoallv displacements so mesh-order is preserved end-to-end.
    std::vector<std::pair<localIdx, localIdx>> procPatchOffset;
    std::vector<int> targetRanks;
    for (auto& boundaryCondition : boundaryConditions_)
    {
        boundaryCondition.correctBoundaryCondition(this->field_);
        if (procPatchCount > 0 && boundaryCondition.patchID() >= firstProcPatch)
        {
            const auto procIdx = boundaryCondition.patchID() - firstProcPatch;
            auto [start, end] = boundaryCondition.range();
            procPatchOffset.emplace_back(start, end);
            targetRanks.push_back(static_cast<int>(nbrRanks[procIdx]));
        }
    }

    if (!procPatchOffset.empty())
    {
        // FIXME dont recompute communication pattern
        auto commPattern = computeCommunicationPattern(this->mesh());
        communicateBoundaryData(
            commPattern, procPatchOffset, targetRanks, this->field_.boundaryData().value()
        );

        // MPI-02 fix: copy received ghost values from boundaryData().value() proc-tail
        // into internalVector() proc-face slots so operators read updated values.
        const auto nIntF = static_cast<localIdx>(this->mesh().nInternalFaces());
        const auto nNonProcBnd = static_cast<localIdx>(bm.offset()[firstProcPatch]);
        const auto nProcBnd = static_cast<localIdx>(bm.nProcBoundaryFaces());

        auto intVecV = this->field_.internalVector().view();
        const auto bndValV = this->field_.boundaryData().value().view();

        parallelFor(
            this->exec(),
            {0, nProcBnd},
            NEON_LAMBDA(const localIdx i) {
                intVecV[nIntF + nNonProcBnd + i] = bndValV[nNonProcBnd + i];
            },
            "syncProcFaceInternalVector"
        );
        fence(this->exec());
    }
}

#define NN_DECLARE_SURFACE_FIELD(TYPENAME) template class SurfaceField<TYPENAME>

NN_FOR_ALL_VALUE_TYPES(NN_DECLARE_SURFACE_FIELD);

}
