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

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp" // these are required for registration

// TODO(operator-registration workaround): drop this once the runtime-selection registry is shared
// across shared objects (e.g. exporting the factory table with default visibility). The operators
// above are declared `extern template`, so simply including the headers does NOT instantiate them
// here and their self-registration only fires in libNeoN. The `_neon` module is compiled
// `-fvisibility=hidden`, giving it a private, empty copy of the factory's lookup table, so scheme
// resolution at assembly time (e.g. "Gauss" for div/laplacian) aborts with "Could not find
// constructor for Gauss". Forcing explicit instantiation here runs the self-registration inside
// `_neon` so its table is populated.
namespace NeoN::finiteVolume::cellCentred
{
template class GaussGreenDiv<scalar>;
template class GaussGreenDiv<Vec3>;
template class GaussGreenDiv<Vec3, scalar>;
template class GaussGreenLaplacian<scalar>;
template class GaussGreenLaplacian<Vec3>;
template class GaussGreenLaplacian<Vec3, scalar>;
} // namespace NeoN::finiteVolume::cellCentred

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
        .def("size", &Expr::size)
        // Resolve each operator's discretisation scheme from a schemes Dictionary
        // (e.g. {"divSchemes": {"div(phi,U)": ["Gauss", "linear"]}}). This is the call
        // that drives the runtime-selection factory lookup (create("Gauss")), so it is
        // also the regression hook for operator self-registration inside _neon.
        .def(
            "read",
            [](Expr& self, const Dictionary& schemes) { self.read(schemes); },
            "schemes"_a,
            "Resolve operator schemes from a Dictionary"
        );
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
    exp_m.def(
        "div",
        [](const ScalarSurfField& flux, const ScalarVolField& phi)
        { return dsl::exp::div(flux, phi); }
    );
    exp_m.def("div", [](const ScalarSurfField& flux) { return dsl::exp::div(flux); });
    exp_m.def("laplacian", &dsl::exp::laplacian<scalar>);
    exp_m.def("laplacian", &dsl::exp::laplacian<Vec3>);
    exp_m.def("grad", &dsl::exp::grad);
    exp_m.def("source", [](ScalarVolField& coeff) { return dsl::exp::source<scalar>(coeff); });
    exp_m.def(
        "source",
        [](const ScalarVolField& coeff, const ScalarVolField& phi)
        { return dsl::exp::source<scalar>(coeff, phi); }
    );

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

    // Registered runtime-selection scheme names per operator factory, keyed by
    // "<operator><value-type>". Regression hook for the force-instantiated operator
    // registration above: an empty list means self-registration did not fire in _neon.
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    m.def(
        "registered_operator_schemes",
        []()
        {
            nb::dict schemes;
            schemes["div<scalar>"] = fvcc::DivOperatorFactory<scalar>::entries();
            schemes["div<Vector>"] = fvcc::DivOperatorFactory<Vec3>::entries();
            schemes["div<Vector,scalar>"] = fvcc::DivOperatorFactory<Vec3, scalar>::entries();
            schemes["laplacian<scalar>"] = fvcc::LaplacianOperatorFactory<scalar>::entries();
            schemes["laplacian<Vector>"] = fvcc::LaplacianOperatorFactory<Vec3>::entries();
            schemes["laplacian<Vector,scalar>"] =
                fvcc::LaplacianOperatorFactory<Vec3, scalar>::entries();
            return schemes;
        }
    );
}

} // namespace NeoN::bindings
