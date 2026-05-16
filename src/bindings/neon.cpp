// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include "NeoN/core/error.hpp"
#include "bindings.hpp"

namespace nb = nanobind;

NB_MODULE(_neon, m)
{
    m.doc() = "NeoN Python bindings";

    // Register NeoNException so C++ NF_THROW produces a catchable Python exception
    static nb::exception<NeoN::NeoNException> neonError(m, "NeoNError", PyExc_RuntimeError);

    // Test helper to verify error handling from Python
    m.def("test_nf_error_exit", []() { NF_THROW("test error from NF_THROW"); });

    // Register all bindings from separate files
    NeoN::bindings::registerExecutors(m);
    NeoN::bindings::registerScalar(m);
    NeoN::bindings::registerVec3(m);
    NeoN::bindings::registerTensor(m);
    NeoN::bindings::registerSymmTensor(m);
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
    NeoN::bindings::registerTensorOps(m);

    // Database bindings
    NeoN::bindings::registerDocument(m);
    NeoN::bindings::registerCollection(m);
    NeoN::bindings::registerDatabase(m);
}
