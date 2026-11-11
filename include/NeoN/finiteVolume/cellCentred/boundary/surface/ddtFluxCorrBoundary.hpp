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

/**
 * @brief Boundary conditions to apply to the density-weighted ddtFluxCorr surfaceScalarField.
 *
 * OpenFOAM evaluates the coupling coefficient of fvc::ddtCorr(rho, U, phi) on the product
 * rho.oldTime()*U.oldTime(), whose patches are 'calculated'. calculatedFvPatchField::fixesValue()
 * is true, so the flux correction is zero on every non-coupled boundary face, whatever the
 * boundary conditions of U are. Coupled (processor) patches keep their own patch type in OpenFOAM,
 * and are not zeroed. Processor patches are the trailing patches of the boundary mesh.
 *
 * @param mesh The mesh the boundary is registered on.
 * @return bcs The surface boundary vector: fixedValue 0 on physical patches, calculated on
 * processor patches.
 */
inline std::vector<SurfaceBoundary<scalar>>
createDensityWeightedFluxCorrBCs(const UnstructuredMesh& mesh)
{
    std::vector<SurfaceBoundary<scalar>> bcs;
    bcs.reserve(mesh.nBoundaries());

    const auto nBoundaries = toLocalIdx(mesh.nBoundaries());
    const localIdx firstProcPatch =
        nBoundaries - toLocalIdx(mesh.boundaryMesh().nProcBoundaryPatches());

    for (localIdx patchID = 0; patchID < nBoundaries; ++patchID)
    {
        Dictionary dict;
        if (patchID < firstProcPatch)
        {
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", scalar(0));
        }
        else
        {
            dict.insert("type", std::string("calculated"));
        }
        bcs.emplace_back(mesh, dict, patchID);
    }

    return bcs;
}

} // namespace NeoN::finiteVolume::cellCentred
