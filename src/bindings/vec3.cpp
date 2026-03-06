// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "NeoN/core/primitives/vec3.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerVec3(nb::module_& m)
{
    // Vec3
    // positions, velocities, normals, and other 3-component quantities.
    nb::class_<NeoN::Vec3>(m, "Vec3", "A 3D vector primitive")
        .def(nb::init<>(), "Create a zero-initialized Vec3")
        .def(
            nb::init<NeoN::scalar, NeoN::scalar, NeoN::scalar>(),
            "x"_a,
            "y"_a,
            "z"_a,
            "Create a Vec3 with specified x, y, z components"
        )
        .def(
            nb::init<NeoN::scalar>(),
            "value"_a,
            "Create a Vec3 with all components set to the same value"
        )

        .def(
            "__getitem__",
            [](const NeoN::Vec3& v, size_t i)
            {
                if (i >= 3) throw std::out_of_range("Vec3 index out of range");
                return v[i];
            },
            "i"_a,
            "Get component by index (0=x, 1=y, 2=z)"
        )
        .def(
            "__setitem__",
            [](NeoN::Vec3& v, size_t i, NeoN::scalar value)
            {
                if (i >= 3) throw std::out_of_range("Vec3 index out of range");
                v[i] = value;
            },
            "i"_a,
            "value"_a,
            "Set component by index"
        )

        .def_prop_rw(
            "x",
            [](const NeoN::Vec3& v) { return v[0]; },
            [](NeoN::Vec3& v, NeoN::scalar val) { v[0] = val; },
            "X component"
        )
        .def_prop_rw(
            "y",
            [](const NeoN::Vec3& v) { return v[1]; },
            [](NeoN::Vec3& v, NeoN::scalar val) { v[1] = val; },
            "Y component"
        )
        .def_prop_rw(
            "z",
            [](const NeoN::Vec3& v) { return v[2]; },
            [](NeoN::Vec3& v, NeoN::scalar val) { v[2] = val; },
            "Z component"
        )

        .def(
            "__add__",
            [](const NeoN::Vec3& a, const NeoN::Vec3& b) { return a + b; },
            "Add two Vec3s"
        )
        .def(
            "__sub__",
            [](const NeoN::Vec3& a, const NeoN::Vec3& b) { return a - b; },
            "Subtract two Vec3s"
        )
        .def(
            "__mul__",
            [](const NeoN::Vec3& v, NeoN::scalar s) { return v * s; },
            "Multiply Vec3 by scalar"
        )
        .def(
            "__rmul__",
            [](const NeoN::Vec3& v, NeoN::scalar s) { return s * v; },
            "Multiply scalar by Vec3"
        )

        .def(
            "__iadd__",
            [](NeoN::Vec3& a, const NeoN::Vec3& b) -> NeoN::Vec3&
            {
                a += b;
                return a;
            }
        )
        .def(
            "__isub__",
            [](NeoN::Vec3& a, const NeoN::Vec3& b) -> NeoN::Vec3&
            {
                a -= b;
                return a;
            }
        )
        .def(
            "__imul__",
            [](NeoN::Vec3& v, NeoN::scalar s) -> NeoN::Vec3&
            {
                v *= s;
                return v;
            }
        )

        .def("__eq__", [](const NeoN::Vec3& a, const NeoN::Vec3& b) { return a == b; })

        .def(
            "dot",
            [](const NeoN::Vec3& a, const NeoN::Vec3& b) { return a & b; },
            "other"_a,
            "Compute dot product with another Vec3"
        )

        .def(
            "mag",
            [](const NeoN::Vec3& v) { return NeoN::mag(v); },
            "Compute magnitude (length) of the vector"
        )

        .def(
            "__repr__",
            [](const NeoN::Vec3& v)
            {
                return "Vec3(" + std::to_string(v[0]) + ", " + std::to_string(v[1]) + ", "
                     + std::to_string(v[2]) + ")";
            }
        )
        .def(
            "__str__",
            [](const NeoN::Vec3& v)
            {
                return "(" + std::to_string(v[0]) + ", " + std::to_string(v[1]) + ", "
                     + std::to_string(v[2]) + ")";
            }
        )
        .def(
            "__len__",
            [](const NeoN::Vec3&) { return 3; },
            "Return the number of components (always 3)"
        );

    m.def(
        "mag",
        static_cast<NeoN::scalar (*)(const NeoN::Vec3&)>(&NeoN::mag),
        "vec"_a,
        "Compute magnitude of a Vec3"
    );

    m.def(
        "dot",
        [](const NeoN::Vec3& a, const NeoN::Vec3& b) { return a & b; },
        "vec1"_a,
        "vec2"_a,
        "Compute dot product of two Vec3s"
    );
}

} // namespace NeoN::bindings
