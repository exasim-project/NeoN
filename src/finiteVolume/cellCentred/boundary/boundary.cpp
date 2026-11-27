// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/boundary.hpp"

namespace NeoN
{

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
