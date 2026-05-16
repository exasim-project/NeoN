// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include "NeoN/finiteVolume/cellCentred/operators/tensorOps.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

void registerTensorOps(nb::module_& m)
{
    m.def(
        "symm",
        [](const fvcc::VolumeField<NeoN::Tensor>& T) { return fvcc::symm(T); },
        "T"_a,
        "Symmetric part of a tensor field: 0.5*(T + T^T)"
    );

    m.def(
        "skew",
        [](const fvcc::VolumeField<NeoN::Tensor>& T) { return fvcc::skew(T); },
        "T"_a,
        "Skew-symmetric part of a tensor field: 0.5*(T - T^T)"
    );

    m.def(
        "mag",
        [](const fvcc::VolumeField<NeoN::Tensor>& T) { return fvcc::mag(T); },
        "T"_a,
        "Frobenius magnitude of a tensor field"
    );

    m.def(
        "mag",
        [](const fvcc::VolumeField<NeoN::SymmTensor>& S) { return fvcc::mag(S); },
        "S"_a,
        "Frobenius magnitude of a symmetric tensor field"
    );

    m.def(
        "dev",
        [](const fvcc::VolumeField<NeoN::SymmTensor>& S) { return fvcc::dev(S); },
        "S"_a,
        "Deviatoric part of a symmetric tensor field"
    );

    m.def(
        "twoSymm",
        [](const fvcc::VolumeField<NeoN::Tensor>& T) { return fvcc::twoSymm(T); },
        "T"_a,
        "Twice the symmetric part of a tensor field: T + T^T"
    );

    m.def(
        "mag",
        [](const fvcc::VolumeField<NeoN::Vec3>& v) { return fvcc::mag(v); },
        "v"_a,
        "Magnitude of a vector field"
    );

    m.def(
        "inner",
        [](const fvcc::VolumeField<NeoN::Vec3>& v1, const fvcc::VolumeField<NeoN::Vec3>& v2)
        { return fvcc::inner(v1, v2); },
        "v1"_a,
        "v2"_a,
        "Inner (dot) product of two vector fields"
    );

    m.def(
        "field_max",
        [](const fvcc::VolumeField<NeoN::scalar>& f, NeoN::scalar val)
        { return fvcc::max(f, val); },
        "f"_a,
        "val"_a,
        "Element-wise max of field and scalar"
    );

    m.def(
        "field_max",
        [](const fvcc::VolumeField<NeoN::scalar>& f1, const fvcc::VolumeField<NeoN::scalar>& f2)
        { return fvcc::max(f1, f2); },
        "f1"_a,
        "f2"_a,
        "Element-wise max of two fields"
    );

    m.def(
        "field_min",
        [](const fvcc::VolumeField<NeoN::scalar>& f, NeoN::scalar val)
        { return fvcc::min(f, val); },
        "f"_a,
        "val"_a,
        "Element-wise min of field and scalar"
    );

    m.def(
        "field_min",
        [](const fvcc::VolumeField<NeoN::scalar>& f1, const fvcc::VolumeField<NeoN::scalar>& f2)
        { return fvcc::min(f1, f2); },
        "f1"_a,
        "f2"_a,
        "Element-wise min of two fields"
    );

    m.def(
        "bound",
        [](fvcc::VolumeField<NeoN::scalar>& f, NeoN::scalar lower) { fvcc::bound(f, lower); },
        "f"_a,
        "lower"_a,
        "Bound field below: f = max(f, lower) — modifies in-place"
    );

    m.def(
        "field_pow",
        [](const fvcc::VolumeField<NeoN::scalar>& f, NeoN::scalar exp)
        { return fvcc::pow(f, exp); },
        "f"_a,
        "exp"_a,
        "Element-wise power: f^exp"
    );

    m.def(
        "dev2",
        [](const fvcc::VolumeField<NeoN::Tensor>& T) { return fvcc::dev2(T); },
        "T"_a,
        "Deviatoric part (2/3 variant) of a tensor field: T - (2/3)*tr(T)*I"
    );

    m.def(
        "dev2",
        [](const fvcc::VolumeField<NeoN::SymmTensor>& S) { return fvcc::dev2(S); },
        "S"_a,
        "Deviatoric part (2/3 variant) of a symmetric tensor field"
    );

    m.def(
        "transpose",
        [](const fvcc::VolumeField<NeoN::Tensor>& T) { return fvcc::transpose(T); },
        "T"_a,
        "Transpose of a tensor field"
    );

    m.def(
        "scalar_tensor_mul",
        [](const fvcc::VolumeField<NeoN::scalar>& s, const fvcc::VolumeField<NeoN::Tensor>& T)
        { return fvcc::mul(s, T); },
        "s"_a,
        "T"_a,
        "Element-wise scalar * tensor field multiplication"
    );

    m.def(
        "tr",
        [](const fvcc::VolumeField<NeoN::Tensor>& T) { return fvcc::tr(T); },
        "T"_a,
        "Trace of a tensor field: tr(T) = T_xx + T_yy + T_zz"
    );

    m.def(
        "devTwoSymm",
        [](const fvcc::VolumeField<NeoN::Tensor>& T) { return fvcc::devTwoSymm(T); },
        "T"_a,
        "devTwoSymm(T) = dev(twoSymm(T)) — deviatoric of twice the symmetric part"
    );

    m.def(
        "doubleInner",
        [](const fvcc::VolumeField<NeoN::Tensor>& T, const fvcc::VolumeField<NeoN::SymmTensor>& S)
        { return fvcc::doubleInner(T, S); },
        "T"_a,
        "S"_a,
        "Double inner product T:S (Frobenius inner product) — returns scalar field"
    );
}

} // namespace NeoN::bindings
