// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerContainerFreeFunctions(nb::module_& m)
{
    // fill for ScalarVector
    m.def(
        "fill",
        [](NeoN::Vector<NeoN::scalar>& cont, NeoN::scalar value) { NeoN::fill(cont, value); },
        "container"_a,
        "value"_a,
        "Fill a ScalarVector with a uniform scalar value"
    );

    m.def(
        "fill_range",
        [](NeoN::Vector<NeoN::scalar>& cont,
           NeoN::scalar value,
           NeoN::localIdx start,
           NeoN::localIdx end) {
            NeoN::fill(cont, value, {start, end});
        },
        "container"_a,
        "value"_a,
        "start"_a,
        "end"_a,
        "Fill a range of a ScalarVector with a uniform scalar value"
    );

    // fill for VectorVector (Vec3)
    m.def(
        "fill",
        [](NeoN::Vector<NeoN::Vec3>& cont, const NeoN::Vec3& value) { NeoN::fill(cont, value); },
        "container"_a,
        "value"_a,
        "Fill a VectorVector with a uniform Vec3 value"
    );

    m.def(
        "fill_range",
        [](NeoN::Vector<NeoN::Vec3>& cont,
           const NeoN::Vec3& value,
           NeoN::localIdx start,
           NeoN::localIdx end) {
            NeoN::fill(cont, value, {start, end});
        },
        "container"_a,
        "value"_a,
        "start"_a,
        "end"_a,
        "Fill a range of a VectorVector with a uniform Vec3 value"
    );

    // fill for LabelVector
    m.def(
        "fill",
        [](NeoN::Vector<NeoN::label>& cont, NeoN::label value) { NeoN::fill(cont, value); },
        "container"_a,
        "value"_a,
        "Fill a LabelVector with a uniform label (integer) value"
    );

    m.def(
        "fill_range",
        [](NeoN::Vector<NeoN::label>& cont,
           NeoN::label value,
           NeoN::localIdx start,
           NeoN::localIdx end) {
            NeoN::fill(cont, value, {start, end});
        },
        "container"_a,
        "value"_a,
        "start"_a,
        "end"_a,
        "Fill a range of a LabelVector with a uniform label value"
    );

    // equal functions for ScalarVector
    m.def(
        "equal",
        [](NeoN::Vector<NeoN::scalar>& cont, NeoN::scalar value)
        { return NeoN::equal(cont, value); },
        "container"_a,
        "value"_a,
        "Check if all elements in a ScalarVector are equal to a value"
    );

    m.def(
        "equal",
        [](const NeoN::Vector<NeoN::scalar>& cont1, const NeoN::Vector<NeoN::scalar>& cont2)
        { return NeoN::equal(cont1, cont2); },
        "container1"_a,
        "container2"_a,
        "Check if two ScalarVectors are element-wise equal"
    );

    // equal functions for VectorVector
    m.def(
        "equal",
        [](NeoN::Vector<NeoN::Vec3>& cont, const NeoN::Vec3& value)
        { return NeoN::equal(cont, value); },
        "container"_a,
        "value"_a,
        "Check if all elements in a VectorVector are equal to a Vec3 value"
    );

    m.def(
        "equal",
        [](const NeoN::Vector<NeoN::Vec3>& cont1, const NeoN::Vector<NeoN::Vec3>& cont2)
        { return NeoN::equal(cont1, cont2); },
        "container1"_a,
        "container2"_a,
        "Check if two VectorVectors are element-wise equal"
    );

    // equal functions for LabelVector
    m.def(
        "equal",
        [](NeoN::Vector<NeoN::label>& cont, NeoN::label value) { return NeoN::equal(cont, value); },
        "container"_a,
        "value"_a,
        "Check if all elements in a LabelVector are equal to a value"
    );

    m.def(
        "equal",
        [](const NeoN::Vector<NeoN::label>& cont1, const NeoN::Vector<NeoN::label>& cont2)
        { return NeoN::equal(cont1, cont2); },
        "container1"_a,
        "container2"_a,
        "Check if two LabelVectors are element-wise equal"
    );
}

} // namespace NeoN::bindings
