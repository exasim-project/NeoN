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

/** @brief Magnitude of a vector field */
VolumeField<scalar> mag(const VolumeField<Vec3>& v);

/** @brief Inner (dot) product of two vector fields */
VolumeField<scalar> inner(const VolumeField<Vec3>& v1, const VolumeField<Vec3>& v2);

/** @brief Element-wise max of field and scalar */
VolumeField<scalar> max(const VolumeField<scalar>& f, scalar val);

/** @brief Element-wise max of two fields */
VolumeField<scalar> max(const VolumeField<scalar>& f1, const VolumeField<scalar>& f2);

/** @brief Element-wise min of field and scalar */
VolumeField<scalar> min(const VolumeField<scalar>& f, scalar val);

/** @brief Element-wise min of two fields */
VolumeField<scalar> min(const VolumeField<scalar>& f1, const VolumeField<scalar>& f2);

/** @brief Bound field below: f = max(f, lower) — modifies in-place */
void bound(VolumeField<scalar>& f, scalar lower);

/** @brief Element-wise power: f^exponent */
VolumeField<scalar> pow(const VolumeField<scalar>& f, scalar exponent);

/** @brief Element-wise scalar * tensor field multiplication */
VolumeField<Tensor> mul(const VolumeField<scalar>& s, const VolumeField<Tensor>& T);

/** @brief Deviatoric part (2/3 variant) of a tensor field: T - (2/3)*tr(T)*I */
VolumeField<Tensor> dev2(const VolumeField<Tensor>& T);

/** @brief Deviatoric part (2/3 variant) of a symmetric tensor field */
VolumeField<SymmTensor> dev2(const VolumeField<SymmTensor>& S);

/** @brief Transpose of a tensor field */
VolumeField<Tensor> transpose(const VolumeField<Tensor>& T);

/** @brief Trace of a tensor field: tr(T) → scalar */
VolumeField<scalar> tr(const VolumeField<Tensor>& T);

/** @brief devTwoSymm(T) = dev(twoSymm(T)) — deviatoric of twice the symmetric part */
VolumeField<SymmTensor> devTwoSymm(const VolumeField<Tensor>& T);

/** @brief Double inner product T:S → scalar (Frobenius inner product) */
VolumeField<scalar> doubleInner(const VolumeField<Tensor>& T, const VolumeField<SymmTensor>& S);

} // namespace NeoN::finiteVolume::cellCentred
