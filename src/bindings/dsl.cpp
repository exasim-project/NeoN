// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>
#include <nanobind/operators.h>

#include "NeoN/dsl/expression.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/dsl/temporalOperator.hpp"
#include "NeoN/dsl/implicit.hpp"
#include "NeoN/dsl/explicit.hpp"
#include "NeoN/dsl/solver.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::dsl
{
template<typename ValueType>
Expression<ValueType> operator+(Expression<ValueType> lhs, const TemporalOperator<ValueType>& rhs)
{
    lhs.addOperator(rhs);
    return lhs;
}
template<typename ValueType>
Expression<ValueType> operator+(const TemporalOperator<ValueType>& lhs, Expression<ValueType> rhs)
{
    rhs.addOperator(lhs);
    return rhs;
}
template<typename ValueType>
Expression<ValueType> operator+(const SpatialOperator<ValueType>& lhs, Expression<ValueType> rhs)
{
    rhs.addOperator(lhs);
    return rhs;
}
template<typename ValueType>
Expression<ValueType> operator-(Expression<ValueType> lhs, const TemporalOperator<ValueType>& rhs)
{
    lhs.addOperator(scalar(-1.0) * rhs);
    return lhs;
}
template<typename ValueType>
Expression<ValueType>
operator-(const TemporalOperator<ValueType>& lhs, const Expression<ValueType>& rhs)
{
    Expression<ValueType> res(lhs.exec());
    res.addOperator(lhs);
    res.addExpression(scalar(-1.0) * rhs);
    return res;
}
template<typename ValueType>
Expression<ValueType>
operator-(const SpatialOperator<ValueType>& lhs, const Expression<ValueType>& rhs)
{
    Expression<ValueType> res(lhs.exec());
    res.addOperator(lhs);
    res.addExpression(scalar(-1.0) * rhs);
    return res;
}
}

namespace NeoN::bindings
{

template<typename ValueType>
void declare_dsl_components(nb::module_& m, const std::string& suffix)
{
    using Expr = dsl::Expression<ValueType>;
    using SpatialOp = dsl::SpatialOperator<ValueType>;
    using TemporalOp = dsl::TemporalOperator<ValueType>;

    nb::class_<SpatialOp>(m, ("SpatialOperator" + suffix).c_str())
        .def("get_name", &SpatialOp::getName)
        .def(scalar() * nb::self)
        .def(nb::self + nb::self) // spatial + spatial -> expression
        .def(nb::self - nb::self) // spatial - spatial -> expression
        // spatial + temporal -> expression
        .def(
            "__add__",
            [](SpatialOp& self, const TemporalOp& rhs) { return self + rhs; },
            nb::is_operator()
        )
        // spatial - temporal -> expression
        .def(
            "__sub__",
            [](SpatialOp& self, const TemporalOp& rhs) { return self - rhs; },
            nb::is_operator()
        );

    nb::class_<TemporalOp>(m, ("TemporalOperator" + suffix).c_str())
        .def("get_name", &TemporalOp::getName)
        .def(scalar() * nb::self)
        .def(nb::self + nb::self) // temporal + temporal -> expression
        .def(nb::self - nb::self) // temporal - temporal -> expression
        // temporal + spatial -> expression
        .def(
            "__add__",
            [](TemporalOp& self, const SpatialOp& rhs) { return self + rhs; },
            nb::is_operator()
        )
        // temporal - spatial -> expression
        .def(
            "__sub__",
            [](TemporalOp& self, const SpatialOp& rhs) { return self - rhs; },
            nb::is_operator()
        );

    nb::class_<Expr>(m, ("Expression" + suffix).c_str())
        .def(nb::init<const Executor&>())
        .def(scalar() * nb::self)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(
            "__add__", [](Expr lhs, const SpatialOp& rhs) { return lhs + rhs; }, nb::is_operator()
        )
        .def(
            "__sub__", [](Expr lhs, const SpatialOp& rhs) { return lhs - rhs; }, nb::is_operator()
        )
        .def(
            "__add__", [](Expr lhs, const TemporalOp& rhs) { return lhs + rhs; }, nb::is_operator()
        )
        .def(
            "__sub__", [](Expr lhs, const TemporalOp& rhs) { return lhs - rhs; }, nb::is_operator()
        )
        .def("size", &Expr::size);
}

void registerDSL(nb::module_& m)
{
    auto exp_m = m.def_submodule("exp", "Explicit operators");
    auto imp_m = m.def_submodule("imp", "Implicit operators");

    declare_dsl_components<scalar>(m, "Scalar");
    declare_dsl_components<Vec3>(m, "Vector");

    using ScalarVolField = NeoN::finiteVolume::cellCentred::VolumeField<scalar>;
    using VectorVolField = NeoN::finiteVolume::cellCentred::VolumeField<Vec3>;
    using ScalarSurfField = NeoN::finiteVolume::cellCentred::SurfaceField<scalar>;

    // Implicit factories
    imp_m.def("ddt", &dsl::imp::ddt<scalar>);
    imp_m.def("ddt", &dsl::imp::ddt<Vec3>);
    imp_m.def("div", &dsl::imp::div<scalar>);
    imp_m.def("div", &dsl::imp::div<Vec3>);
    imp_m.def("laplacian", &dsl::imp::laplacian<scalar>);
    imp_m.def("laplacian", &dsl::imp::laplacian<Vec3>);
    imp_m.def("source", &dsl::imp::source<scalar>);
    imp_m.def("source", &dsl::imp::source<Vec3>);

    // Explicit factories
    exp_m.def("ddt", &dsl::exp::ddt<scalar>);
    exp_m.def("ddt", &dsl::exp::ddt<Vec3>);
    exp_m.def("div", nb::overload_cast<const ScalarSurfField&, ScalarVolField&>(&dsl::exp::div));
    exp_m.def("div", nb::overload_cast<const ScalarSurfField&>(&dsl::exp::div));
    exp_m.def(
        "laplacian",
        nb::overload_cast<const ScalarSurfField&, ScalarVolField&>(&dsl::exp::laplacian)
    );
    exp_m.def(
        "laplacian",
        nb::overload_cast<const ScalarSurfField&, VectorVolField&>(&dsl::exp::laplacian)
    );
    exp_m.def("grad", &dsl::exp::grad);
    exp_m.def("source", &dsl::exp::source);

    // solve
    m.def(
        "solve",
        [](dsl::Expression<scalar>& exp,
           ScalarVolField& sol,
           scalar t,
           scalar dt,
           const Dictionary& schemes,
           const Dictionary& solution) { return dsl::solve(exp, sol, t, dt, schemes, solution); }
    );

    m.def(
        "solve",
        [](dsl::Expression<Vec3>& exp,
           VectorVolField& sol,
           scalar t,
           scalar dt,
           const Dictionary& schemes,
           const Dictionary& solution) { return dsl::solve(exp, sol, t, dt, schemes, solution); }
    );
}

} // namespace NeoN::bindings
