// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/core/primitives/symmTensor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include "boundary/volume/empty.hpp"
#include "boundary/volume/calculated.hpp"
#include "boundary/volume/extrapolated.hpp"
#include "boundary/volume/fixedValue.hpp"
#include "boundary/volume/fixedGradient.hpp"
#include "boundary/volume/fixedFluxPressure.hpp"
#include "boundary/volume/symmetry.hpp"
#include "boundary/volume/slip.hpp"

#include "boundary/surface/empty.hpp"
#include "boundary/surface/calculated.hpp"
#include "boundary/surface/fixedValue.hpp"
#include "boundary/surface/symmetry.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief creates a vector of boundary conditions of type calculated for every boundary
 *
 * @tparam Type of the Boundary ie SurfaceBoundary<scalar>
 */
template<typename BoundaryType>
std::vector<BoundaryType> createCalculatedBCs(const UnstructuredMesh& mesh)
{
    std::vector<BoundaryType> bcs;
    bcs.reserve(static_cast<std::size_t>(mesh.nBoundaries()));

    for (localIdx patchID = 0; patchID < mesh.nBoundaries(); patchID++)
    {
        Dictionary patchDict({{"type", std::string("calculated")}});
        bcs.emplace_back(mesh, patchDict, patchID);
    }
    return bcs;
};

template<typename BoundaryType>
std::vector<BoundaryType> createCalculatedProcBCs(const UnstructuredMesh& mesh)
{
    std::vector<BoundaryType> bcs;
    bcs.reserve(static_cast<std::size_t>(mesh.nBoundaries()));
    // Distributed-aware 'calculated': physical patches get the calculated BC (the value is computed
    // by an operator), but processor (coupled) patches must carry the processor halo-exchange BC so
    // that a single correctBoundaryConditions() fills their tail with the NEIGHBOUR cell value
    // across the rank boundary. createCalculatedBCs leaves processor patches 'calculated' (no
    // exchange), so a derived field that needs the neighbour value at proc faces — e.g. the cell
    // gradient consumed by the corrected/limitedCorrected face-normal gradient — must use this
    // variant and then correctBoundaryConditions() instead of a bespoke halo exchange.
    // Processor (coupled) patches are the trailing patches of the boundary mesh;
    // nProcBoundaryPatches is 0 on a serial / non-distributed mesh, so this degenerates to
    // createCalculatedBCs there.
    const auto nProcPatches = mesh.boundaryMesh().nProcBoundaryPatches();
    const auto firstProcPatch = mesh.nBoundaries() - nProcPatches;
    for (localIdx patchID = 0; patchID < mesh.nBoundaries(); patchID++)
    {
        const std::string type =
            (patchID >= firstProcPatch) ? std::string("processor") : std::string("calculated");
        Dictionary patchDict({{"type", type}});
        bcs.emplace_back(mesh, patchDict, patchID);
    }
    return bcs;
};

template<typename BoundaryType>
std::vector<BoundaryType> createExtrapolatedBCs(const UnstructuredMesh& mesh)
{
    std::vector<BoundaryType> bcs;
    bcs.reserve(mesh.nBoundaries());
    // Processor (coupled) patches are the trailing patches of the boundary mesh. A coupled patch
    // must always carry the processor halo-exchange BC so its boundary tail holds the NEIGHBOUR
    // cell value — interpolation/flux at processor faces read that tail as the far-side value.
    // An 'extrapolated' tail would instead hold the OWNER value, which silently degenerates
    // linear interpolation at proc faces to w*own+(1-w)*own=own (different on each rank) and so
    // makes the assembled pressure matrix non-symmetric. Only physical patches get 'extrapolated'.
    const auto nProcPatches = mesh.boundaryMesh().nProcBoundaryPatches();
    const auto firstProcPatch = mesh.nBoundaries() - nProcPatches;
    for (localIdx patchID = 0; patchID < mesh.nBoundaries(); patchID++)
    {
        const std::string type =
            (patchID >= firstProcPatch) ? std::string("processor") : std::string("extrapolated");
        Dictionary patchDict({{"type", type}});
        bcs.emplace_back(mesh, patchDict, patchID);
    }
    return bcs;
};

}

namespace NeoN
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::VolumeBoundaryFactory<scalar>;
template class fvcc::VolumeBoundaryFactory<Vec3>;
template class fvcc::VolumeBoundaryFactory<Tensor>;
template class fvcc::VolumeBoundaryFactory<SymmTensor>;

template class fvcc::volumeBoundary::FixedValue<scalar>;
template class fvcc::volumeBoundary::FixedValue<Vec3>;

template class fvcc::volumeBoundary::FixedGradient<scalar>;
template class fvcc::volumeBoundary::FixedGradient<Vec3>;

template class fvcc::volumeBoundary::FixedFluxPressure<scalar>;

template class fvcc::volumeBoundary::Calculated<scalar>;
template class fvcc::volumeBoundary::Calculated<Vec3>;
template class fvcc::volumeBoundary::Calculated<Tensor>;
template class fvcc::volumeBoundary::Calculated<SymmTensor>;

template class fvcc::volumeBoundary::Extrapolated<scalar>;
template class fvcc::volumeBoundary::Extrapolated<Vec3>;

template class fvcc::volumeBoundary::Empty<scalar>;
template class fvcc::volumeBoundary::Empty<Vec3>;

template class fvcc::volumeBoundary::Symmetry<scalar>;
template class fvcc::volumeBoundary::Symmetry<Vec3>;

template class fvcc::volumeBoundary::Slip<scalar>;
template class fvcc::volumeBoundary::Slip<Vec3>;

template class fvcc::SurfaceBoundaryFactory<scalar>;
template class fvcc::SurfaceBoundaryFactory<Vec3>;
template class fvcc::SurfaceBoundaryFactory<Tensor>;
template class fvcc::SurfaceBoundaryFactory<SymmTensor>;

template class fvcc::surfaceBoundary::FixedValue<scalar>;
template class fvcc::surfaceBoundary::FixedValue<Vec3>;

template class fvcc::surfaceBoundary::Calculated<scalar>;
template class fvcc::surfaceBoundary::Calculated<Vec3>;
template class fvcc::surfaceBoundary::Calculated<Tensor>;
template class fvcc::surfaceBoundary::Calculated<SymmTensor>;

template class fvcc::surfaceBoundary::Empty<scalar>;
template class fvcc::surfaceBoundary::Empty<Vec3>;

template class fvcc::surfaceBoundary::Symmetry<scalar>;
template class fvcc::surfaceBoundary::Symmetry<Vec3>;

}
