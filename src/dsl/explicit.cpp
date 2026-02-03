// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
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
    return SpatialOperator<NeoN::scalar>(
        fvcc::LaplacianOperator<NeoN::scalar>(dsl::Operator::Type::Explicit, gamma, phi)
    );
}

SpatialOperator<Vec3>
laplacian(const fvcc::SurfaceField<NeoN::scalar>& gamma, fvcc::VolumeField<NeoN::Vec3>& phi)
{
    return SpatialOperator<Vec3>(fvcc::LaplacianOperator(dsl::Operator::Type::Explicit, gamma, phi)
    );
}

SpatialOperator<NeoN::Vec3> grad(fvcc::VolumeField<NeoN::scalar>& phi)
{
    return SpatialOperator<NeoN::Vec3>(
        fvcc::GradOperator<NeoN::Vec3>(dsl::Operator::Type::Explicit, phi)
    );
}

SpatialOperator<scalar>
source(fvcc::VolumeField<NeoN::scalar>& coeff, fvcc::VolumeField<NeoN::scalar>& phi)
{
    return SpatialOperator<scalar>(fvcc::SourceTerm(dsl::Operator::Type::Explicit, coeff, phi));
}

SpatialOperator<scalar> sourceU(fvcc::VolumeField<NeoN::scalar>& coeff)
{
    return SpatialOperator<scalar>(fvcc::SourceUTerm(dsl::Operator::Type::Explicit, coeff));
}

SpatialOperator<Vec3> viscousStress(
    const fvcc::SurfaceField<scalar>& nuF,
    const fvcc::SurfaceField<scalar>& nutF,
    const fvcc::SurfaceField<scalar>& nuTildeF,
    const fvcc::VolumeField<Vec3>& U,
    const fvcc::VolumeField<Vec3>& gradUx,
    const fvcc::VolumeField<Vec3>& gradUy,
    const fvcc::VolumeField<Vec3>& gradUz
)
{
    return SpatialOperator<Vec3>(fvcc::ViscousStressOperator(
        dsl::Operator::Type::Explicit, nuF, nutF, nuTildeF, U, gradUx, gradUy, gradUz
    ));
}

} // namespace NeoN
