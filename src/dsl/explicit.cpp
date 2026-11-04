// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/dsl/explicit.hpp"

namespace NeoN::dsl::exp
{
namespace fvcc = NeoN::finiteVolume::cellCentred;

SpatialOperator<scalar> div(const fvcc::SurfaceField<NeoN::scalar>& flux)
{
    return SpatialOperator<scalar>(fvcc::SurfaceIntegrate<scalar>(flux));
}

SpatialOperator<NeoN::Vec3> grad(const fvcc::VolumeField<NeoN::scalar>& phi)
{
    return SpatialOperator<NeoN::Vec3>(
        fvcc::GradOperator<NeoN::Vec3>(dsl::Operator::Type::Explicit, phi)
    );
}


} // namespace NeoN
