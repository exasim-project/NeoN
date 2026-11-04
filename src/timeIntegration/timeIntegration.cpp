// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/timeIntegration/timeIntegration.hpp"
#include "NeoN/timeIntegration/forwardEuler.hpp"
#include "NeoN/timeIntegration/backwardEuler.hpp"
#include "NeoN/timeIntegration/steadyState.hpp"

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
