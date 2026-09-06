// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{
/**
 * @brief Boundary condition to apply to ddtFluxCorr surfaceScalarField.
 * It sets the flux correction to zero for all fixedValue velocity boundaries
 * by calling .correctBoundaryConditions().
 *
 * @param mesh The mesh the boundary is registered on.
 * @param u The velocity field to check the boundary conditions on.
 * @return bcs The surface boundary vector with updated BC types.
 */
inline std::vector<SurfaceBoundary<scalar>>
createFluxCorrBCsFromU(const UnstructuredMesh& mesh, const VolumeField<Vec3>& u)
{
    std::vector<SurfaceBoundary<scalar>> bcs;
    const auto& uBCs = u.boundaryConditions();

    bcs.reserve(uBCs.size());

    for (auto patchID = 0u; patchID < uBCs.size(); ++patchID)
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
