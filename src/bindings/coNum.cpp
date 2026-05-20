// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>

#include "NeoN/finiteVolume/cellCentred/auxiliary/coNum.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerCoNum(nb::module_& m)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    m.def(
        "compute_co_num",
        [](const fvcc::SurfaceField<NeoN::scalar>& faceFlux, NeoN::scalar dt
        ) -> std::pair<NeoN::scalar, NeoN::scalar> { return fvcc::computeCoNum(faceFlux, dt); },
        "face_flux"_a,
        "dt"_a,
        "Compute (max, mean) Courant number from a scalar face-flux SurfaceField."
    );
}

} // namespace NeoN::bindings
