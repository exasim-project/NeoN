// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/dsl/temporalOperator.hpp"
#include "NeoN/dsl/ddt.hpp"

// TODO: decouple from fvcc
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceTerm.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN::dsl::imp
{

// TODO all arguments could be const
template<typename ValueType>
TemporalOperator<ValueType> ddt(fvcc::VolumeField<ValueType>& phi)
{
    return fvcc::DdtOperator(dsl::Operator::Type::Implicit, phi);
}

// Density-weighted temporal operator ddt(rho, phi): diagonal uses rho, rhs uses
// oldTime(rho), giving the conservative (rho_n*phi - rho_o*phi_o)/dt form.
template<typename ValueType>
TemporalOperator<ValueType> ddt(fvcc::VolumeField<scalar>& rho, fvcc::VolumeField<ValueType>& phi)
{
    return fvcc::DdtOperator<ValueType>(dsl::Operator::Type::Implicit, rho, phi);
}

template<typename ValueType>
SpatialOperator<ValueType>
source(fvcc::VolumeField<scalar>& coeff, fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(fvcc::SourceTerm(dsl::Operator::Type::Implicit, coeff, phi));
}

// SuSp: sign-aware implicit source — max(coeff, 0) on the diagonal, min(coeff, 0)
// explicitly to the rhs. Keeps the matrix diagonally dominant for a coefficient of
// either sign (e.g. the kOmegaSST cross-diffusion term).
template<typename ValueType>
SpatialOperator<ValueType> susp(fvcc::VolumeField<scalar>& coeff, fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(
        fvcc::SourceTerm(dsl::Operator::Type::Implicit, coeff, phi, true)
    );
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

} // namespace NeoN
