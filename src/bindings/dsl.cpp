// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/variant.h>
#include <nanobind/operators.h>

#include "NeoN/dsl/expression.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/dsl/temporalOperator.hpp"
#include "NeoN/dsl/implicit.hpp"
#include "NeoN/dsl/explicit.hpp"
#include "NeoN/dsl/solver.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "bindings.hpp"

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/operators/boundedDiv.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/interpolation/linearUpwind.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/corrected.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/limitedCorrected.hpp" // these are required for registration

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
template class BoundedDiv<scalar>;
template class BoundedDiv<Vec3>;
template class BoundedDiv<Vec3, scalar>;
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

    // Implicit factories. ddt is overloaded (single-field and density-weighted), so bind
    // via lambdas that resolve the overload by arity rather than taking &ddt<T> directly.
    imp_m.def("ddt", [](ScalarVolField& phi) { return dsl::imp::ddt<scalar>(phi); });
    imp_m.def("ddt", [](VectorVolField& phi) { return dsl::imp::ddt<Vec3>(phi); });
    // Density-weighted ddt(rho, U) overload (two field args).
    imp_m.def(
        "ddt",
        [](ScalarVolField& rho, VectorVolField& phi) { return dsl::imp::ddt<Vec3>(rho, phi); }
    );
    imp_m.def("div", &dsl::imp::div<scalar>);
    imp_m.def("div", &dsl::imp::div<Vec3>);
    imp_m.def("laplacian", &dsl::imp::laplacian<scalar>);
    imp_m.def("laplacian", &dsl::imp::laplacian<Vec3>);
    imp_m.def("source", &dsl::imp::source<scalar>);
    imp_m.def("source", &dsl::imp::source<Vec3>);
    imp_m.def("susp", &dsl::imp::susp<scalar>);
    imp_m.def("susp", &dsl::imp::susp<Vec3>);

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
    // Vec3 explicit Su source — wraps a reconstructed cell body force as an rhs operator.
    exp_m.def("source", [](VectorVolField& coeff) { return dsl::exp::source<Vec3>(coeff); });
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

    // Test hook: assemble only the implicit SPATIAL operators of an expression into an empty
    // linear system for `mesh` and hand back the host-side matrix values and rhs. There is no
    // solver-independent way to observe assembly from Python otherwise, so binding tests for
    // implicit factories (in particular the imp.susp sign split, where the same operator name
    // and Python type covers both the Sp and the Su branch) would only be able to check that the
    // overload resolves, not what it assembles.
    m.def(
        "assemble_spatial",
        [](dsl::Expression<scalar>& expr, const UnstructuredMesh& mesh)
        {
            auto ls = la::createEmptyLinearSystem<scalar>(mesh);
            expr.assembleSpatialOperator(ls);
            auto lsHost = ls.copyToHost();
            const auto values = lsHost.matrix().values().view();
            const auto rhs = lsHost.rhs().view();
            std::vector<scalar> v(values.begin(), values.end());
            std::vector<scalar> r(rhs.begin(), rhs.end());
            return std::make_pair(std::move(v), std::move(r));
        },
        "expr"_a,
        "mesh"_a,
        "Assemble the implicit spatial operators of a scalar expression; returns (values, rhs)"
    );
    m.def(
        "assemble_spatial",
        [](dsl::Expression<Vec3>& expr, const UnstructuredMesh& mesh)
        {
            auto ls = la::createEmptyLinearSystem<Vec3>(mesh);
            expr.assembleSpatialOperator(ls);
            auto lsHost = ls.copyToHost();
            const auto values = lsHost.matrix().values().view();
            const auto rhs = lsHost.rhs().view();
            std::vector<Vec3> v(values.begin(), values.end());
            std::vector<Vec3> r(rhs.begin(), rhs.end());
            return std::make_pair(std::move(v), std::move(r));
        },
        "expr"_a,
        "mesh"_a,
        "Assemble the implicit spatial operators of a vector expression; returns (values, rhs)"
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
            // Surface-interpolation schemes register from their headers rather than through the
            // explicit instantiations above, but they are listed here for the same reason: a
            // failed lookup aborts the process (NF_ERROR_EXIT), so a Python test cannot probe for
            // a missing scheme by catching an exception -- it has to read the table instead.
            schemes["surfaceInterpolation<scalar>"] =
                fvcc::SurfaceInterpolationFactory<scalar>::entries();
            schemes["surfaceInterpolation<Vector>"] =
                fvcc::SurfaceInterpolationFactory<Vec3>::entries();
            return schemes;
        }
    );
}

} // namespace NeoN::bindings
