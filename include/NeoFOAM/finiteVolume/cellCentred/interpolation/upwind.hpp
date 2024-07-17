// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/mesh/stencil/fvccGeometryScheme.hpp"

#include "Kokkos_Core.hpp"

#include <functional>

namespace NeoFOAM::finiteVolume::cellCentred
{

class Upwind : public SurfaceInterpolationFactory::Register<Upwind>
{

public:

    Upwind(const Executor& exec, const UnstructuredMesh& mesh);


    static std::string name() { return "upwind"; }

    static std::string doc() { return "upwind interpolation"; }

    static std::string schema() { return "none"; }

    void interpolate(
        fvcc::SurfaceField<scalar>& surfaceField, const fvcc::VolumeField<scalar>& volField
    ) override;

    void interpolate(
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& volField
    ) override;

    std::unique_ptr<SurfaceInterpolationFactory> clone() const override;


private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoFOAM
