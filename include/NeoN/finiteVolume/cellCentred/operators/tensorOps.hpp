// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/** @brief Symmetric part of a tensor field: 0.5*(T + T^T) */
VolumeField<SymmTensor> symm(const VolumeField<Tensor>& T);

/** @brief Skew-symmetric part of a tensor field: 0.5*(T - T^T) */
VolumeField<Tensor> skew(const VolumeField<Tensor>& T);

/** @brief Frobenius magnitude of a tensor field */
VolumeField<scalar> mag(const VolumeField<Tensor>& T);

/** @brief Frobenius magnitude of a symmetric tensor field */
VolumeField<scalar> mag(const VolumeField<SymmTensor>& S);

/** @brief Deviatoric part of a symmetric tensor field: S - tr(S)/3 * I */
VolumeField<SymmTensor> dev(const VolumeField<SymmTensor>& S);

/** @brief Twice the symmetric part of a tensor field: T + T^T */
VolumeField<SymmTensor> twoSymm(const VolumeField<Tensor>& T);

} // namespace NeoN::finiteVolume::cellCentred
