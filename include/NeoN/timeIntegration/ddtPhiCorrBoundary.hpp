// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

inline std::vector<SurfaceBoundary<scalar>>
createPhiCorrBCsFromU(const UnstructuredMesh& mesh, const VolumeField<Vec3>& U)
{
    std::vector<SurfaceBoundary<scalar>> bcs;
    const auto& uBCs = U.boundaryConditions();

    bcs.reserve(uBCs.size());

    for (localIdx patchID = 0; patchID < (localIdx)uBCs.size(); ++patchID)
    {
        const auto attrs = uBCs[patchID].attributes();
        Dictionary dict;

        if (attrs.fixesValue)
        {
            // Zero correction on U fixedValue patches
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", scalar(0));
            bcs.emplace_back(mesh, dict, patchID);
        }
        else
        {
            dict.insert("type", std::string("calculated"));
            bcs.emplace_back(mesh, dict, patchID);
        }
    }

    return bcs;
}

} // namespace NeoN::finiteVolume::cellCentred
