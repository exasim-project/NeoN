// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"

namespace NeoN::turbulenceModels
{
using VolVectorField = NeoN::finiteVolume::cellCentred::VolumeField<Vec3>;
using VolScalarField = NeoN::finiteVolume::cellCentred::VolumeField<scalar>;
class ReynoldsStress
{
public:

    ReynoldsStress(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        Input input =
            Dictionary {{"surfaceInterpolation", "linear"}, {"faceNormalGradient", "uncorrected"}}
    );

    VolVectorField divDevReff(const VolVectorField& velocity, const VolScalarField& nuEff) const;

private:

    Executor exec_;
    const UnstructuredMesh& mesh_;
    Input input_;
    finiteVolume::cellCentred::SurfaceInterpolation<scalar> surfaceInterpolation_;
    finiteVolume::cellCentred::GaussGreenLaplacian<Vec3> laplacian_;
};

} // namespace NeoN::turbulenceModels
