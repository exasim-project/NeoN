// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Compiled instead of ginkgoSolve.cpp when NeoN_WITH_GINKGO is OFF, so module.cpp stays
// unconditional and every symbol still exists, raising on first use.

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

    m.def(
        "_la_matrix_solve",
        [](nb::args, nb::kwargs) -> nb::dict { throw std::runtime_error("built without Ginkgo"); }
    );
    m.def(
        "_la_matrix_probe",
        [](nb::args, nb::kwargs) -> nb::dict { throw std::runtime_error("built without Ginkgo"); }
    );
    m.def(
        "_la_system_solve",
        [](nb::args, nb::kwargs) -> nb::dict { throw std::runtime_error("built without Ginkgo"); }
    );
    m.def(
        "_la_system_probe",
        [](nb::args, nb::kwargs) -> nb::dict { throw std::runtime_error("built without Ginkgo"); }
    );
    m.def(
        "_la_stored_diagonal",
        [](nb::args, nb::kwargs) { throw std::runtime_error("built without Ginkgo"); }
    );

    // linear_algebra surface, registered so `import blockamr.linear_algebra` works.
    m.def(
        "la_laplacian",
        [](nb::args, nb::kwargs) { throw std::runtime_error("built without Ginkgo"); }
    );

    struct MatrixStub
    {
    };
    struct OperatorStub
    {
    };
    struct LinearSystemStub
    {
    };
    struct LaSolverStub
    {
    };
    nb::class_<MatrixStub>(m, "Matrix")
        .def(
            "__init__",
            [](MatrixStub*, nb::args, nb::kwargs)
            { throw std::runtime_error("built without Ginkgo"); }
        );
    nb::class_<OperatorStub>(m, "Operator");
    nb::class_<LinearSystemStub>(m, "LinearSystem")
        .def(
            "__init__",
            [](LinearSystemStub*, nb::args, nb::kwargs)
            { throw std::runtime_error("built without Ginkgo"); }
        );
    nb::class_<LaSolverStub>(m, "Solver")
        .def(
            "__init__",
            [](LaSolverStub*, nb::args, nb::kwargs)
            { throw std::runtime_error("built without Ginkgo"); }
        );

    m.def("profile_report", []() -> nb::dict { return nb::dict(); });
    m.def("profile_reset", []() {});
}
