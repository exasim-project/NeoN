// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/dsl/explicit.hpp"

namespace NeoN::dsl::exp
{
namespace fvcc = NeoN::finiteVolume::cellCentred;

SpatialOperator<scalar>
div(const fvcc::SurfaceField<NeoN::scalar>& faceFlux, fvcc::VolumeField<NeoN::scalar>& phi)
{
    return SpatialOperator<scalar>(fvcc::DivOperator(dsl::Operator::Type::Explicit, faceFlux, phi));
}

SpatialOperator<scalar> div(const fvcc::SurfaceField<NeoN::scalar>& flux)
{
    return SpatialOperator<scalar>(fvcc::SurfaceIntegrate<scalar>(flux));
}

SpatialOperator<scalar>
laplacian(const fvcc::SurfaceField<NeoN::scalar>& gamma, fvcc::VolumeField<NeoN::scalar>& phi)
{
    return SpatialOperator<scalar>(
        fvcc::LaplacianOperator(dsl::Operator::Type::Explicit, gamma, phi)
    );
}

SpatialOperator<Vec3>
laplacian(const fvcc::SurfaceField<NeoN::scalar>& gamma, fvcc::VolumeField<NeoN::Vec3>& phi)
{
    return SpatialOperator<Vec3>(fvcc::LaplacianOperator(dsl::Operator::Type::Explicit, gamma, phi)
    );
}

SpatialOperator<scalar>
source(fvcc::VolumeField<NeoN::scalar>& coeff, fvcc::VolumeField<NeoN::scalar>& phi)
{
    return SpatialOperator<scalar>(fvcc::SourceTerm(dsl::Operator::Type::Explicit, coeff, phi));
}

} // namespace NeoN
