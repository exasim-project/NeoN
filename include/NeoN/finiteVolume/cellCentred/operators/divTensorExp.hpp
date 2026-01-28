// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/dsl/spatialOperator.hpp"

namespace NeoN::finiteVolume::cellCentred
{

void computeDivNuDev2TGradUExp(
    const VolumeField<scalar>& nu,
    const VolumeField<Vec3>& gradUx,
    const VolumeField<Vec3>& gradUy,
    const VolumeField<Vec3>& gradUz,
    const SurfaceInterpolation<Vec3>& surfInterpVec,
    Vector<Vec3>& rhs,
    const dsl::Coeff operatorScaling
);

void computeLaplacianScalarGammaVectorExp(
    const FaceNormalGradient<Vec3>& faceNormalGradient, // snGrad scheme
    const SurfaceField<scalar>& gammaF,                 // e.g. nut on faces (all faces!)
    const VolumeField<Vec3>& U,                         // volVectorField
    Vector<Vec3>& lapU,                                 // cell vector result
    const dsl::Coeff operatorScaling
);

} // namespace NeoN::finiteVolume::cellCentred
