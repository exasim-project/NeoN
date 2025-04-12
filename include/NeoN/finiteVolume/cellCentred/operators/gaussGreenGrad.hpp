// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

class GaussGreenGrad
{
public:

    GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh);

    // fvcc::VolumeField<Vector> grad(const fvcc::VolumeField<scalar>& phi);

    void grad(const VolumeField<scalar>& phi, VolumeField<Vector>& gradPhi);

    VolumeField<Vector> grad(const VolumeField<scalar>& phi);

private:

    const UnstructuredMesh& mesh_;
    SurfaceInterpolation<scalar> surfaceInterpolation_;
};

} // namespace NeoFOAM
