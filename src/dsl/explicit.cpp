// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/dsl/explicit.hpp"

#include <variant>

#include "NeoN/core/logging.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gradOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"

namespace NeoN
{
namespace finiteVolume
{
namespace cellCentred
{
template<typename ValueType>
class SurfaceField;
template<typename ValueType>
class VolumeField;
} // namespace cellCentred
} // namespace finiteVolume
} // namespace NeoN

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
