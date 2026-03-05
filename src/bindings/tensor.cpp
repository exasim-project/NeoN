// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "NeoN/core/primitives/tensor.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerTensor(nb::module_& m)
{
    nb::class_<NeoN::Tensor>(m, "Tensor", "A 3x3 tensor primitive (row-major)")
        .def(nb::init<>(), "Create a zero-initialized Tensor")
        .def(
            nb::init<
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar,
                NeoN::scalar>(),
            "xx"_a,
            "xy"_a,
            "xz"_a,
            "yx"_a,
            "yy"_a,
            "yz"_a,
            "zx"_a,
            "zy"_a,
            "zz"_a,
            "Create a Tensor with specified components"
        )
        .def(
            nb::init<NeoN::scalar>(),
            "value"_a,
            "Create a Tensor with all components set to the same value"
        )

        .def(
            "__getitem__",
            [](const NeoN::Tensor& t, size_t i)
            {
                if (i >= 9) throw std::out_of_range("Tensor index out of range");
                return t[i];
            },
            "i"_a,
            "Get component by linear index"
        )
        .def(
            "__setitem__",
            [](NeoN::Tensor& t, size_t i, NeoN::scalar value)
            {
                if (i >= 9) throw std::out_of_range("Tensor index out of range");
                t[i] = value;
            },
            "i"_a,
            "value"_a,
            "Set component by linear index"
        )

        .def_prop_ro("xx", &NeoN::Tensor::xx, "XX component")
        .def_prop_ro("xy", &NeoN::Tensor::xy, "XY component")
        .def_prop_ro("xz", &NeoN::Tensor::xz, "XZ component")
        .def_prop_ro("yx", &NeoN::Tensor::yx, "YX component")
        .def_prop_ro("yy", &NeoN::Tensor::yy, "YY component")
        .def_prop_ro("yz", &NeoN::Tensor::yz, "YZ component")
        .def_prop_ro("zx", &NeoN::Tensor::zx, "ZX component")
        .def_prop_ro("zy", &NeoN::Tensor::zy, "ZY component")
        .def_prop_ro("zz", &NeoN::Tensor::zz, "ZZ component")

        .def(
            "__add__",
            [](const NeoN::Tensor& a, const NeoN::Tensor& b) { return a + b; },
            "Add two Tensors"
        )
        .def(
            "__sub__",
            [](const NeoN::Tensor& a, const NeoN::Tensor& b) { return a - b; },
            "Subtract two Tensors"
        )
        .def(
            "__mul__",
            [](const NeoN::Tensor& t, NeoN::scalar s) { return t * s; },
            "Multiply Tensor by scalar"
        )
        .def(
            "__rmul__",
            [](const NeoN::Tensor& t, NeoN::scalar s) { return s * t; },
            "Multiply scalar by Tensor"
        )
        .def(
            "__truediv__",
            [](const NeoN::Tensor& t, NeoN::scalar s) { return t / s; },
            "Divide Tensor by scalar"
        )

        .def("__eq__", [](const NeoN::Tensor& a, const NeoN::Tensor& b) { return a == b; })

        .def(
            "mag",
            [](const NeoN::Tensor& t) { return NeoN::mag(t); },
            "Compute Frobenius norm"
        )

        .def(
            "__repr__",
            [](const NeoN::Tensor& t)
            {
                return "Tensor(" + std::to_string(t[0]) + ", " + std::to_string(t[1]) + ", "
                     + std::to_string(t[2]) + ", " + std::to_string(t[3]) + ", "
                     + std::to_string(t[4]) + ", " + std::to_string(t[5]) + ", "
                     + std::to_string(t[6]) + ", " + std::to_string(t[7]) + ", "
                     + std::to_string(t[8]) + ")";
            }
        )
        .def(
            "__len__",
            [](const NeoN::Tensor&) { return 9; },
            "Return the number of components (always 9)"
        );
}

} // namespace NeoN::bindings
