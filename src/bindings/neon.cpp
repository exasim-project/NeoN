// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include "bindings.hpp"

namespace nb = nanobind;

NB_MODULE(_neon, m)
{
    m.doc() = "NeoN Python bindings";

    // Register all bindings from separate files
    NeoN::bindings::registerExecutors(m);
    NeoN::bindings::registerScalar(m);
    NeoN::bindings::registerVec3(m);
    NeoN::bindings::registerVectors(m);
    NeoN::bindings::registerContainerFreeFunctions(m);
    NeoN::bindings::registerBoundaryMesh(m);
    NeoN::bindings::registerUnstructuredMesh(m);
    NeoN::bindings::registerSurfaceField(m);
    NeoN::bindings::registerVolumeField(m);
    NeoN::bindings::registerSurfaceInterpolation(m);
    NeoN::bindings::registerInputs(m);
    NeoN::bindings::registerCoNum(m);
    NeoN::bindings::registerInitialization(m);
    NeoN::bindings::registerLinearAlgebra(m);
    NeoN::bindings::registerDSL(m);

    // Database bindings
    NeoN::bindings::registerDocument(m);
    NeoN::bindings::registerCollection(m);
    NeoN::bindings::registerDatabase(m);
}
