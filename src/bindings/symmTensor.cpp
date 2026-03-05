// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "NeoN/core/primitives/symmTensor.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerSymmTensor(nb::module_& m)
{
    nb::class_<NeoN::SymmTensor>(m, "SymmTensor", "A symmetric 3x3 tensor primitive")
        .def(nb::init<>(), "Create a zero-initialized SymmTensor")
        .def(
            nb::init<
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar>(),
            "xx"_a,
            "xy"_a,
            "xz"_a,
            "yy"_a,
            "yz"_a,
            "zz"_a,
            "Create a SymmTensor with specified components"
        )
        .def(
            nb::init<NeoN::scalar>(),
            "value"_a,
            "Create a SymmTensor with all components set to the same value"
        )

        .def(
            "__getitem__",
            [](const NeoN::SymmTensor& s, size_t i)
            {
                if (i >= 6) throw std::out_of_range("SymmTensor index out of range");
                return s[i];
            },
            "i"_a,
            "Get component by index"
        )
        .def(
            "__setitem__",
            [](NeoN::SymmTensor& s, size_t i, NeoN::scalar value)
            {
                if (i >= 6) throw std::out_of_range("SymmTensor index out of range");
                s[i] = value;
            },
            "i"_a,
            "value"_a,
            "Set component by index"
        )

        .def_prop_ro("xx", &NeoN::SymmTensor::xx, "XX component")
        .def_prop_ro("xy", &NeoN::SymmTensor::xy, "XY component")
        .def_prop_ro("xz", &NeoN::SymmTensor::xz, "XZ component")
        .def_prop_ro("yy", &NeoN::SymmTensor::yy, "YY component")
        .def_prop_ro("yz", &NeoN::SymmTensor::yz, "YZ component")
        .def_prop_ro("zz", &NeoN::SymmTensor::zz, "ZZ component")

        .def(
            "__add__",
            [](const NeoN::SymmTensor& a, const NeoN::SymmTensor& b) { return a + b; },
            "Add two SymmTensors"
        )
        .def(
            "__sub__",
            [](const NeoN::SymmTensor& a, const NeoN::SymmTensor& b) { return a - b; },
            "Subtract two SymmTensors"
        )
        .def(
            "__mul__",
            [](const NeoN::SymmTensor& s, NeoN::scalar v) { return s * v; },
            "Multiply SymmTensor by scalar"
        )
        .def(
            "__rmul__",
            [](const NeoN::SymmTensor& s, NeoN::scalar v) { return v * s; },
            "Multiply scalar by SymmTensor"
        )
        .def(
            "__truediv__",
            [](const NeoN::SymmTensor& s, NeoN::scalar v) { return s / v; },
            "Divide SymmTensor by scalar"
        )

        .def("__eq__", [](const NeoN::SymmTensor& a, const NeoN::SymmTensor& b) { return a == b; })

        .def(
            "mag",
            [](const NeoN::SymmTensor& s) { return NeoN::mag(s); },
            "Compute Frobenius norm"
        )

        .def(
            "__repr__",
            [](const NeoN::SymmTensor& s)
            {
                return "SymmTensor(" + std::to_string(s[0]) + ", " + std::to_string(s[1]) + ", "
                     + std::to_string(s[2]) + ", " + std::to_string(s[3]) + ", "
                     + std::to_string(s[4]) + ", " + std::to_string(s[5]) + ")";
            }
        )
        .def(
            "__len__",
            [](const NeoN::SymmTensor&) { return 6; },
            "Return the number of components (always 6)"
        );
}

} // namespace NeoN::bindings
