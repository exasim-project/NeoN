// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Stub compiled instead of ginkgo_solve.cpp when NeoN_WITH_GINKGO is OFF
// (selected in CMakeLists.txt) so module.cpp stays unconditional.

#include <nanobind/nanobind.h>

#include <stdexcept>

#include "bindings.hpp"

namespace nb = nanobind;

void registerGinkgoSolve(nb::module_& m)
{
    m.def(
        "ginkgo_solve",
        [](nb::args, nb::kwargs) -> nb::dict { throw std::runtime_error("built without Ginkgo"); }
    );
}
