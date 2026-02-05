// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{
struct TensorVecField
{
    VolumeField<Vec3> Tx;
    VolumeField<Vec3> Ty;
    VolumeField<Vec3> Tz;
};
}
