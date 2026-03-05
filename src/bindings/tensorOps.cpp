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
}

} // namespace NeoN::bindings
