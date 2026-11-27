// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/boundary.hpp"

#include "NeoN/finiteVolume/cellCentred/boundary/surface/calculated.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/empty.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/fixedValue.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/symmetry.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/calculated.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/empty.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/extrapolated.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/fixedFluxPressure.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/fixedGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/fixedValue.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/inletOutlet.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/outletInlet.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/slip.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/symmetry.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/totalPressure.hpp"

namespace NeoN
{
class SymmTensor;
class Tensor;

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
template class fvcc::volumeBoundary::InletOutlet<scalar>;
template class fvcc::volumeBoundary::InletOutlet<Vec3>;
template class fvcc::volumeBoundary::OutletInlet<scalar>;
template class fvcc::volumeBoundary::OutletInlet<Vec3>;
template class fvcc::volumeBoundary::TotalPressure<scalar>;
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
