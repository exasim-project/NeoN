// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/dictionary.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include "boundary/volume/empty.hpp"
#include "boundary/volume/calculated.hpp"
#include "boundary/volume/extrapolated.hpp"
#include "boundary/volume/fixedValue.hpp"
#include "boundary/volume/fixedGradient.hpp"

#include "boundary/surface/empty.hpp"
#include "boundary/surface/calculated.hpp"
#include "boundary/surface/fixedValue.hpp"

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
    for (localIdx patchID = 0; patchID < mesh.nBoundaries(); patchID++)
    {
        Dictionary patchDict({{"type", std::string("calculated")}});
        bcs.push_back(BoundaryType(mesh, patchDict, patchID));
    }
    return bcs;
};

template<typename BoundaryType>
std::vector<BoundaryType> createExtrapolatedBCs(const UnstructuredMesh& mesh)
{
    std::vector<BoundaryType> bcs;
    for (localIdx patchID = 0; patchID < mesh.nBoundaries(); patchID++)
    {
        Dictionary patchDict({{"type", std::string("extrapolated")}});
        bcs.push_back(BoundaryType(mesh, patchDict, patchID));
    }
    return bcs;
};

}

namespace NeoN
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::VolumeBoundaryFactory<scalar>;
template class fvcc::VolumeBoundaryFactory<Vec3>;

template class fvcc::volumeBoundary::FixedValue<scalar>;
template class fvcc::volumeBoundary::FixedValue<Vec3>;

template class fvcc::volumeBoundary::FixedGradient<scalar>;
template class fvcc::volumeBoundary::FixedGradient<Vec3>;

template class fvcc::volumeBoundary::Calculated<scalar>;
template class fvcc::volumeBoundary::Calculated<Vec3>;

template class fvcc::volumeBoundary::Extrapolated<scalar>;
template class fvcc::volumeBoundary::Extrapolated<Vec3>;

template class fvcc::volumeBoundary::Empty<scalar>;
template class fvcc::volumeBoundary::Empty<Vec3>;

template class fvcc::SurfaceBoundaryFactory<scalar>;
template class fvcc::SurfaceBoundaryFactory<Vec3>;

template class fvcc::surfaceBoundary::FixedValue<scalar>;
template class fvcc::surfaceBoundary::FixedValue<Vec3>;

template class fvcc::surfaceBoundary::Calculated<scalar>;
template class fvcc::surfaceBoundary::Calculated<Vec3>;

template class fvcc::surfaceBoundary::Empty<scalar>;
template class fvcc::surfaceBoundary::Empty<Vec3>;

}
