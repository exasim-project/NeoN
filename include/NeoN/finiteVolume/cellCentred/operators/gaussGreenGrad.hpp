// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoN::finiteVolume::cellCentred
{

class GaussGreenGrad
{
public:

    GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh);

    // fvcc::VolumeField<Vec3> grad(const fvcc::VolumeField<scalar>& phi);

    void grad(const VolumeField<scalar>& phi, VolumeField<Vec3>& gradPhi);

    VolumeField<Vec3> grad(const VolumeField<scalar>& phi);

private:

    const UnstructuredMesh& mesh_;
    SurfaceInterpolation<scalar> surfaceInterpolation_;
};

} // namespace NeoN
