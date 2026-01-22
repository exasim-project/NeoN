// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "NeoN/core/primitives/scalar.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerScalar(nb::module_& m)
{
    // Scalar type information
    // NeoN::scalar is either float or double depending on NeoN_DP_SCALAR compile flag
    m.attr("ROOTVSMALL") = NeoN::ROOTVSMALL;

    // Scalar mag function
    m.def(
        "scalar_mag",
        [](NeoN::scalar s) { return NeoN::mag(s); },
        "s"_a,
        "Compute the absolute value (magnitude) of a scalar"
    );

    // one<scalar> and zero<scalar> traits
    m.def(
        "scalar_one", []() { return NeoN::one<NeoN::scalar>(); }, "Return the scalar value 1.0"
    );

    m.def(
        "scalar_zero", []() { return NeoN::zero<NeoN::scalar>(); }, "Return the scalar value 0.0"
    );

    // Expose whether we're using double precision
#ifdef NeoN_DP_SCALAR
    m.attr("DOUBLE_PRECISION") = true;
#else
    m.attr("DOUBLE_PRECISION") = false;
#endif
}

} // namespace NeoN::bindings
