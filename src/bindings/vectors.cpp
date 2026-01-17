// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
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

template<typename T>
nb::class_<NeoN::Vector<T>>
declare_vector(nb::module_& m, const std::string& name, const std::string& doc = "")
{
    return nb::class_<NeoN::Vector<T>>(m, name.c_str(), doc.c_str())
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx>(),
            "exec"_a,
            "size"_a,
            "Create an uninitialized vector of given size on an executor"
        )
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx, T>(),
            "exec"_a,
            "size"_a,
            "value"_a,
            "Create a vector with uniform value"
        )
        .def(
            nb::init<const NeoN::Executor&, std::vector<T>>(),
            "exec"_a,
            "values"_a,
            "Create a vector from a Python list"
        )
        .def("size", &NeoN::Vector<T>::size, "Get the size of the vector")
        .def("empty", &NeoN::Vector<T>::empty, "Check if the vector is empty")
        .def("__len__", &NeoN::Vector<T>::size, "Get the size (for Python len())")
        .def("exec", &NeoN::Vector<T>::exec, "Get the executor associated with this vector")
        .def(
            "copy_to_host",
            static_cast<NeoN::Vector<T> (NeoN::Vector<T>::*)() const>(&NeoN::Vector<T>::copyToHost),
            "Copy the vector data to host (CPU) and return as new vector"
        )
        .def("resize", &NeoN::Vector<T>::resize, "size"_a, "Resize the vector to a new size")
        .def(
            "__repr__",
            [name](const NeoN::Vector<T>& v)
            { return "<" + name + " size=" + std::to_string(v.size()) + ">"; }
        )
        .def(
            "__str__",
            [name](const NeoN::Vector<T>& v)
            { return name + "(size=" + std::to_string(v.size()) + ")"; }
        );
}

void registerVectors(nb::module_& m)
{
    // Vector T
    // execution spaces (CPU, GPU). It's similar to std::vector but with support
    // for heterogeneous computing via executors.

    // Vector<scalar> - for scalar fields
    declare_vector<NeoN::scalar>(
        m, "ScalarVector", "A vector of scalar values with executor support"
    );

    // Vector<Vec3> - for vector fields
    declare_vector<NeoN::Vec3>(m, "VectorVector", "A vector of Vec3 values with executor support");

    // Vector<label> - for integer indices
    declare_vector<NeoN::label>(
        m, "LabelVector", "A vector of label (integer) values with executor support"
    );
}

} // namespace NeoN::bindings
