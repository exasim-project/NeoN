// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

// TODO we should get rid of this include since it includes details
// from a general implementation
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceTerm.hpp"


namespace NeoN::dsl::exp
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

SpatialOperator<scalar>
div(const fvcc::SurfaceField<NeoN::scalar>& faceFlux, fvcc::VolumeField<NeoN::scalar>& phi);

SpatialOperator<scalar> div(const fvcc::SurfaceField<NeoN::scalar>& flux);

SpatialOperator<scalar>
source(fvcc::VolumeField<NeoN::scalar>& coeff, fvcc::VolumeField<NeoN::scalar>& phi);

} // namespace NeoN
