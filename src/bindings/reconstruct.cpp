// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "NeoN/finiteVolume/cellCentred/operators/reconstruct.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerReconstruct(nb::module_& m)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    m.def(
        "reconstruct",
        &fvcc::reconstruct,
        "ssf"_a,
        "Reconstruct a cell vector field from a surface scalar flux (fvc::reconstruct)."
    );
}

} // namespace NeoN::bindings
