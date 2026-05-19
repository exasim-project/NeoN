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

    // Proc patches sit at the tail of uBCs (createCalculatedBCs /
    // createExtrapolatedBCs / readVolBoundaryConditions all place proc
    // patches after the regular ones). For ddtCorr the boundary at a
    // proc face must be a `processor` BC so that
    // SurfaceField::correctBoundaryConditions runs the Processor BC's
    // `correctBoundaryCondition` and copies internalVector[proc-tail] into
    // boundaryData.value()[proc-tail] (the dual-storage invariant — see
    // project memory `project_surface_field_dual_storage`). Without that
    // copy, ddtCorr.boundaryData().value() stays at default zero, and
    // SurfaceField operator* (which multiplies BOTH internalVector AND
    // boundaryData() element-wise) produces a result with zero
    // boundaryData proc-tail — corrupting downstream consumers.
    const auto procPatchCount = mesh.boundaryMesh().nProcBoundaryPatches();
    const auto firstProcPatch =
        static_cast<unsigned>(uBCs.size()) - static_cast<unsigned>(procPatchCount);

    for (auto patchID = 0u; patchID < uBCs.size(); ++patchID)
    {
        Dictionary dict;

        if (patchID >= firstProcPatch && procPatchCount > 0)
        {
            // Processor patch — must be a `processor` SurfaceBoundary so
            // its correctBoundaryCondition syncs internalVector ->
            // boundaryData proc-tail.
            dict.insert("type", std::string("processor"));
            bcs.emplace_back(mesh, dict, patchID);
            continue;
        }

        const auto attrs = uBCs[patchID].attributes();
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
