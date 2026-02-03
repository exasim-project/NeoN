// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/dsl/temporalOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

// TODO we should get rid of this include since it includes details
// from a general implementation
#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gradOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceTerm.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceUTerm.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/viscousStressOperator.hpp"

namespace NeoN::dsl::exp
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

template<typename ValueType>
TemporalOperator<ValueType> ddt(fvcc::VolumeField<ValueType>& phi)
{
    return fvcc::DdtOperator(dsl::Operator::Type::Explicit, phi);
}

SpatialOperator<scalar>
div(const fvcc::SurfaceField<scalar>& faceFlux, fvcc::VolumeField<scalar>& phi);

SpatialOperator<scalar> div(const fvcc::SurfaceField<scalar>& flux);

SpatialOperator<scalar>
laplacian(const fvcc::SurfaceField<scalar>& gamma, fvcc::VolumeField<scalar>& phi);


SpatialOperator<Vec3>
laplacian(const fvcc::SurfaceField<scalar>& gamma, fvcc::VolumeField<Vec3>& phi);

SpatialOperator<Vec3> grad(fvcc::VolumeField<scalar>& phi);

SpatialOperator<scalar> source(fvcc::VolumeField<scalar>& coeff, fvcc::VolumeField<scalar>& phi);

SpatialOperator<scalar> sourceU(fvcc::VolumeField<scalar>& coeff);

SpatialOperator<Vec3> viscousStress(
    const fvcc::SurfaceField<scalar>& nuF,
    const fvcc::SurfaceField<scalar>& nutF,
    const fvcc::SurfaceField<scalar>& nuTildeF,
    const fvcc::VolumeField<Vec3>& U,
    const fvcc::VolumeField<Vec3>& gradUx,
    const fvcc::VolumeField<Vec3>& gradUy,
    const fvcc::VolumeField<Vec3>& gradUz
);

} // namespace NeoN
