// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/vector/vectorTypeDefs.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerVectors(nb::module_& m)
{
    // Vector T
    // execution spaces (CPU, GPU). It's similar to std::vector but with support
    // for heterogeneous computing via executors.

    // Vector<scalar> - for scalar fields
    nb::class_<NeoN::Vector<NeoN::scalar>>(
        m, "ScalarVector", "A vector of scalar values with executor support"
    )
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx>(),
            "exec"_a,
            "size"_a,
            "Create an uninitialized ScalarVector of given size on an executor"
        )
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx, NeoN::scalar>(),
            "exec"_a,
            "size"_a,
            "value"_a,
            "Create a ScalarVector with uniform value"
        )
        .def(
            nb::init<const NeoN::Executor&, std::vector<NeoN::scalar>>(),
            "exec"_a,
            "values"_a,
            "Create a ScalarVector from a Python list"
        )

        .def("size", &NeoN::Vector<NeoN::scalar>::size, "Get the size of the vector")
        .def("empty", &NeoN::Vector<NeoN::scalar>::empty, "Check if the vector is empty")
        .def("__len__", &NeoN::Vector<NeoN::scalar>::size, "Get the size (for Python len())")

        .def(
            "exec",
            &NeoN::Vector<NeoN::scalar>::exec,
            "Get the executor associated with this vector"
        )

        .def(
            "copy_to_host",
            static_cast<NeoN::Vector<NeoN::scalar> (NeoN::Vector<NeoN::scalar>::*)() const>(
                &NeoN::Vector<NeoN::scalar>::copyToHost
            ),
            "Copy the vector data to host (CPU) and return as new vector"
        )

        .def(
            "resize",
            &NeoN::Vector<NeoN::scalar>::resize,
            "size"_a,
            "Resize the vector to a new size"
        )

        .def(
            "__repr__",
            [](const NeoN::Vector<NeoN::scalar>& v)
            { return "<ScalarVector size=" + std::to_string(v.size()) + ">"; }
        )
        .def(
            "__str__",
            [](const NeoN::Vector<NeoN::scalar>& v)
            { return "ScalarVector(size=" + std::to_string(v.size()) + ")"; }
        );

    // Vector<Vec3> - for vector fields
    nb::class_<NeoN::Vector<NeoN::Vec3>>(
        m, "VectorVector", "A vector of Vec3 values with executor support"
    )
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx>(),
            "exec"_a,
            "size"_a,
            "Create an uninitialized VectorVector of given size"
        )
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx, NeoN::Vec3>(),
            "exec"_a,
            "size"_a,
            "value"_a,
            "Create a VectorVector with uniform value"
        )
        .def(
            nb::init<const NeoN::Executor&, std::vector<NeoN::Vec3>>(),
            "exec"_a,
            "values"_a,
            "Create a VectorVector from a Python list"
        )
        .def("size", &NeoN::Vector<NeoN::Vec3>::size)
        .def("empty", &NeoN::Vector<NeoN::Vec3>::empty)
        .def("__len__", &NeoN::Vector<NeoN::Vec3>::size)
        .def("exec", &NeoN::Vector<NeoN::Vec3>::exec)
        .def(
            "copy_to_host",
            static_cast<NeoN::Vector<NeoN::Vec3> (NeoN::Vector<NeoN::Vec3>::*)() const>(
                &NeoN::Vector<NeoN::Vec3>::copyToHost
            )
        )
        .def("resize", &NeoN::Vector<NeoN::Vec3>::resize, "size"_a)
        .def(
            "__repr__",
            [](const NeoN::Vector<NeoN::Vec3>& v)
            { return "<VectorVector size=" + std::to_string(v.size()) + ">"; }
        );

    // Vector<label> - for integer indices
    nb::class_<NeoN::Vector<NeoN::label>>(
        m, "LabelVector", "A vector of label (integer) values with executor support"
    )
        .def(nb::init<const NeoN::Executor&, NeoN::localIdx>(), "exec"_a, "size"_a)
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx, NeoN::label>(),
            "exec"_a,
            "size"_a,
            "value"_a
        )
        .def(nb::init<const NeoN::Executor&, std::vector<NeoN::label>>(), "exec"_a, "values"_a)
        .def("size", &NeoN::Vector<NeoN::label>::size)
        .def("empty", &NeoN::Vector<NeoN::label>::empty)
        .def("__len__", &NeoN::Vector<NeoN::label>::size)
        .def("exec", &NeoN::Vector<NeoN::label>::exec)
        .def(
            "copy_to_host",
            static_cast<NeoN::Vector<NeoN::label> (NeoN::Vector<NeoN::label>::*)() const>(
                &NeoN::Vector<NeoN::label>::copyToHost
            )
        )
        .def("resize", &NeoN::Vector<NeoN::label>::resize, "size"_a)
        .def(
            "__repr__",
            [](const NeoN::Vector<NeoN::label>& v)
            { return "<LabelVector size=" + std::to_string(v.size()) + ">"; }
        );
}

} // namespace NeoN::bindings
