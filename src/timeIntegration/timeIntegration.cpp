// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/timeIntegration/timeIntegration.hpp"
#include "NeoN/timeIntegration/forwardEuler.hpp"
#include "NeoN/timeIntegration/backwardEuler.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN::timeIntegration
{

template class ForwardEuler<fvcc::VolumeField<scalar>>;

template class BackwardEuler<fvcc::VolumeField<scalar>>;

} // namespace NeoN::dsl
