// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include "NeoN/finiteVolume/cellCentred/operators/mules.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerMules(nb::module_& m)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    m.def(
        "mules_explicit_solve",
        &fvcc::mulesExplicitSolve,
        "alpha"_a,
        "phi"_a,
        "alpha_phi"_a,
        "delta_t"_a,
        "psi_max"_a = 1.0,
        "psi_min"_a = 0.0,
        "n_limiter_iter"_a = 3,
        "Bounded explicit MULES (FCT) solve (MULES::explicitSolve, simplified path: "
        "rho=1, Sp=Su=0, static mesh). Limits alpha_phi in place and advances alpha "
        "conservatively (no clamp)."
    );
}

} // namespace NeoN::bindings
