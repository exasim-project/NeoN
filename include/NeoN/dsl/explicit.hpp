// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/dsl/temporalOperator.hpp"
#include "NeoN/dsl/ddt.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
// TODO: decouple from fvcc
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceTerm.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gradOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"

namespace NeoN::dsl::exp
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

template<typename ValueType>
TemporalOperator<ValueType> ddt(fvcc::VolumeField<ValueType>& phi)
{
    return fvcc::DdtOperator(dsl::Operator::Type::Explicit, phi);
}

template<typename ValueType>
SpatialOperator<ValueType>
source(const fvcc::VolumeField<scalar>& coeff, const fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(fvcc::SourceTerm(dsl::Operator::Type::Explicit, coeff, phi));
}

template<typename ValueType>
SpatialOperator<ValueType>
div(const fvcc::SurfaceField<scalar>& faceFlux, const fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(
        fvcc::DivOperator(dsl::Operator::Type::Explicit, faceFlux, phi)
    );
}

SpatialOperator<scalar> div(const fvcc::SurfaceField<scalar>& flux);

template<typename ValueType>
SpatialOperator<ValueType>
laplacian(const fvcc::SurfaceField<scalar>& gamma, fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(
        fvcc::LaplacianOperator<ValueType>(dsl::Operator::Type::Explicit, gamma, phi)
    );
}

SpatialOperator<Vec3> grad(const fvcc::VolumeField<scalar>& phi);

template<typename ValueType>
SpatialOperator<ValueType> source(fvcc::VolumeField<ValueType>& coeff)
{
    return SpatialOperator<ValueType>(fvcc::SourceTerm(dsl::Operator::Type::Explicit, coeff));
}

} // namespace NeoN
