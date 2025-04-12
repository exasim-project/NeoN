// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/surfaceField.hpp"

// TODO we should get rid of this include since it includes details
// from a general implementation
#include "NeoFOAM/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/sourceTerm.hpp"


namespace NeoFOAM::dsl::exp
{

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

SpatialOperator<scalar>
div(const fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux, fvcc::VolumeField<NeoFOAM::scalar>& phi);

SpatialOperator<scalar> div(const fvcc::SurfaceField<NeoFOAM::scalar>& flux);

SpatialOperator<scalar>
source(fvcc::VolumeField<NeoFOAM::scalar>& coeff, fvcc::VolumeField<NeoFOAM::scalar>& phi);

} // namespace NeoFOAM
