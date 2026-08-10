// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <sstream>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/core/primitives/tensor.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

namespace
{

void checkIndex(size_t i, size_t j)
{
    if (i >= 3 || j >= 3) throw std::out_of_range("Tensor index out of range");
}

} // namespace

void registerTensor(nb::module_& m)
{
    // Tensor
    // 3x3 tensor primitive, e.g. the velocity gradient grad(U). Registering it as a
    // nanobind class is what makes the Vector<Tensor> (TensorVector) constructors that
    // take a Tensor - the uniform value and the list of values overloads - callable.
    nb::class_<NeoN::Tensor>(m, "Tensor", "A 3x3 tensor primitive (row major)")
        .def(nb::init<>(), "Create a zero-initialized Tensor")
        .def(
            nb::init<NeoN::scalar>(),
            "diag"_a,
            "Create a diagonal tensor, i.e. diag * the identity tensor"
        )
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
            "t00"_a,
            "t01"_a,
            "t02"_a,
            "t10"_a,
            "t11"_a,
            "t12"_a,
            "t20"_a,
            "t21"_a,
            "t22"_a,
            "Create a Tensor from its nine components in row major order"
        )
        .def(
            "__init__",
            [](NeoN::Tensor& self, const std::vector<NeoN::scalar>& values)
            {
                if (values.size() != 9)
                {
                    throw std::invalid_argument(
                        "Tensor expects 9 components in row major order, got "
                        + std::to_string(values.size())
                    );
                }
                new (&self) NeoN::Tensor();
                for (size_t k = 0; k < 9; ++k)
                {
                    self.data()[k] = values[k];
                }
            },
            "values"_a,
            "Create a Tensor from a flat sequence of 9 components in row major order"
        )

        .def(
            "__getitem__",
            [](const NeoN::Tensor& t, std::pair<size_t, size_t> ij)
            {
                checkIndex(ij.first, ij.second);
                return t(ij.first, ij.second);
            },
            "ij"_a,
            "Get component by (row, column) index"
        )
        .def(
            "__setitem__",
            [](NeoN::Tensor& t, std::pair<size_t, size_t> ij, NeoN::scalar value)
            {
                checkIndex(ij.first, ij.second);
                t(ij.first, ij.second) = value;
            },
            "ij"_a,
            "value"_a,
            "Set component by (row, column) index"
        )

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
            "__iadd__",
            [](NeoN::Tensor& a, const NeoN::Tensor& b) -> NeoN::Tensor&
            {
                a += b;
                return a;
            }
        )
        .def(
            "__isub__",
            [](NeoN::Tensor& a, const NeoN::Tensor& b) -> NeoN::Tensor&
            {
                a -= b;
                return a;
            }
        )
        .def(
            "__imul__",
            [](NeoN::Tensor& t, NeoN::scalar s) -> NeoN::Tensor&
            {
                t *= s;
                return t;
            }
        )
        .def("__eq__", [](const NeoN::Tensor& a, const NeoN::Tensor& b) { return a == b; })

        .def(
            "trace",
            [](const NeoN::Tensor& t) { return t.trace(); },
            "Compute the trace, i.e. the sum of the diagonal"
        )
        .def(
            "transpose", [](const NeoN::Tensor& t) { return t.T(); }, "Return the transposed tensor"
        )
        .def(
            "dot",
            [](const NeoN::Tensor& t, const NeoN::Vec3& v) { return t & v; },
            "vec"_a,
            "Compute the matrix vector product T . v"
        )
        .def(
            "mag",
            [](const NeoN::Tensor& t) { return NeoN::mag(t); },
            "Compute the Frobenius norm of the tensor"
        )

        .def(
            "__len__",
            [](const NeoN::Tensor&) { return 9; },
            "Return the number of components (always 9)"
        )
        .def(
            "__array__",
            [](NeoN::Tensor& self)
            {
                size_t shape[2] {3, 3};
                return nb::ndarray<NeoN::scalar, nb::numpy, nb::c_contig>(self.data(), 2, shape);
            },
            nb::rv_policy::reference_internal,
            "Get a (3, 3) numpy view on the tensor components"
        )
        .def(
            "__repr__",
            [](const NeoN::Tensor& t)
            {
                std::ostringstream oss;
                oss << "Tensor" << t;
                return oss.str();
            }
        )
        .def(
            "__str__",
            [](const NeoN::Tensor& t)
            {
                std::ostringstream oss;
                oss << t;
                return oss.str();
            }
        );

    m.def(
        "mag",
        [](const NeoN::Tensor& t) { return NeoN::mag(t); },
        "tensor"_a,
        "Compute the Frobenius norm of a Tensor"
    );
}

} // namespace NeoN::bindings
