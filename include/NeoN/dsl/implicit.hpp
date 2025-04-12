// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/dsl/temporalOperator.hpp"
#include "NeoFOAM/dsl/ddt.hpp"

// TODO: decouple from fvcc
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/ddtOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/sourceTerm.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::dsl::imp
{


template<typename ValueType>
TemporalOperator<ValueType> ddt(fvcc::VolumeField<ValueType>& phi)
{
    return fvcc::DdtOperator(dsl::Operator::Type::Implicit, phi);
}

template<typename ValueType>
SpatialOperator<ValueType>
source(fvcc::VolumeField<scalar>& coeff, fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(fvcc::SourceTerm(dsl::Operator::Type::Implicit, coeff, phi));
}

template<typename ValueType>
SpatialOperator<ValueType>
div(fvcc::SurfaceField<scalar>& faceFlux, fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(
        fvcc::DivOperator(dsl::Operator::Type::Implicit, faceFlux, phi)
    );
}

template<typename ValueType>
SpatialOperator<ValueType>
laplacian(fvcc::SurfaceField<scalar>& gamma, fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(
        fvcc::LaplacianOperator<ValueType>(dsl::Operator::Type::Implicit, gamma, phi)
    );
}

} // namespace NeoFOAM
