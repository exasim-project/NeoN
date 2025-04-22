// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

// TODO we should get rid of this include since it includes details
// from a general implementation
#include "NeoN/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceTerm.hpp"


namespace NeoN::dsl::exp
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

SpatialOperator<scalar>
div(const fvcc::SurfaceField<scalar>& faceFlux, fvcc::VolumeField<scalar>& phi);

SpatialOperator<scalar> div(const fvcc::SurfaceField<scalar>& flux);

SpatialOperator<scalar>
laplacian(const fvcc::SurfaceField<scalar>& gamma, fvcc::VolumeField<scalar>& phi);

SpatialOperator<scalar> source(fvcc::VolumeField<scalar>& coeff, fvcc::VolumeField<scalar>& phi);

} // namespace NeoN
