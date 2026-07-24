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
    m.def(
        "ginkgo_solve_composite",
        [](nb::args, nb::kwargs) -> nb::dict { throw std::runtime_error("built without Ginkgo"); }
    );
    m.def(
        "ginkgo_solve_face_coeffs",
        [](nb::args, nb::kwargs) -> nb::dict { throw std::runtime_error("built without Ginkgo"); }
    );

    struct FaceCoeffSolverStub
    {
    };
    struct FaceCoeffCsrSolverStub
    {
    };
    nb::class_<FaceCoeffSolverStub>(m, "FaceCoeffSolver")
        .def(
            "__init__",
            [](FaceCoeffSolverStub*, nb::args, nb::kwargs)
            { throw std::runtime_error("built without Ginkgo"); }
        );
    nb::class_<FaceCoeffCsrSolverStub>(m, "FaceCoeffCsrSolver")
        .def(
            "__init__",
            [](FaceCoeffCsrSolverStub*, nb::args, nb::kwargs)
            { throw std::runtime_error("built without Ginkgo"); }
        );

    m.def("profile_report", []() -> nb::dict { return nb::dict(); });
    m.def("profile_reset", []() {});
}
