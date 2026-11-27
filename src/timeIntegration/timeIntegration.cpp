// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/timeIntegration/timeIntegration.hpp"

#include "NeoN/timeIntegration/forwardEuler.hpp"
#include "NeoN/timeIntegration/backwardEuler.hpp"
#include "NeoN/timeIntegration/steadyState.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN::timeIntegration
{

template class ForwardEuler<fvcc::VolumeField<scalar>>;
template class ForwardEuler<fvcc::VolumeField<Vec3>>;

template class BackwardEuler<fvcc::VolumeField<scalar>>;
template class BackwardEuler<fvcc::VolumeField<Vec3>>;

template class SteadyState<fvcc::VolumeField<scalar>>;
template class SteadyState<fvcc::VolumeField<Vec3>>;

} // namespace NeoN::dsl
