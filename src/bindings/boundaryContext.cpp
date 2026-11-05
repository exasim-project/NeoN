// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerBoundaryContext(nb::module_& m)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    // Lets a Python solver pass the same per-call auxiliary state (face flux, old-time
    // field, deltaT, simulation time, ...) a C++ caller passes via
    // field.correctBoundaryConditions(ctx) -- without it, a Python solver can only call the
    // no-context overload, where every flux-/time-dependent BC (inletOutlet, totalPressure,
    // waveTransmissive, pressureInletOutletVelocity, a sine-forced fixedValue, ...)
    // degrades to its documented no-context fallback.
    nb::class_<fvcc::BoundaryContext>(m, "BoundaryContext")
        .def(nb::init<>())
        .def(
            "insert",
            nb::overload_cast<std::string, const fvcc::VolumeField<NeoN::scalar>&>(
                &fvcc::BoundaryContext::insert
            ),
            "name"_a,
            "field"_a,
            // Index 1 = self (ctx), 2 = name, 3 = field for a bound method -- ties the
            // field's Python lifetime to ctx's, since BoundaryContext stores a raw,
            // non-owning pointer to it (see the class doc).
            nb::keep_alive<1, 3>(),
            "Insert a named scalar volume field"
        )
        .def(
            "insert",
            nb::overload_cast<std::string, const fvcc::VolumeField<NeoN::Vec3>&>(
                &fvcc::BoundaryContext::insert
            ),
            "name"_a,
            "field"_a,
            nb::keep_alive<1, 3>(),
            "Insert a named Vec3 volume field"
        )
        .def(
            "insert",
            nb::overload_cast<std::string, const fvcc::SurfaceField<NeoN::scalar>&>(
                &fvcc::BoundaryContext::insert
            ),
            "name"_a,
            "field"_a,
            nb::keep_alive<1, 3>(),
            "Insert a named scalar surface field (e.g. the face flux phi)"
        )
        .def(
            "insert_value",
            &fvcc::BoundaryContext::insertValue,
            "name"_a,
            "value"_a,
            "Insert a named plain scalar value (e.g. the current deltaT or simulation time)"
        );
}

} // namespace NeoN::bindings
