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
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGradVec3.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDivTensor.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"

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
    declare_dsl_components<Tensor>(m, "Tensor");

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
    imp_m.def(
        "source",
        [](fvcc::VolumeField<scalar>& coeff, fvcc::VolumeField<scalar>& phi, bool susp)
        {
            return dsl::SpatialOperator<scalar>(
                fvcc::SourceTerm<scalar>(dsl::Operator::Type::Implicit, coeff, phi, susp)
            );
        },
        "coeff"_a,
        "phi"_a,
        "susp"_a = false,
        "Implicit source term. susp=True enables SuSp mode (positive→diagonal, negative→source)"
    );
    imp_m.def(
        "source",
        [](fvcc::VolumeField<scalar>& coeff, fvcc::VolumeField<Vec3>& phi, bool susp)
        {
            return dsl::SpatialOperator<Vec3>(
                fvcc::SourceTerm<Vec3>(dsl::Operator::Type::Implicit, coeff, phi, susp)
            );
        },
        "coeff"_a,
        "phi"_a,
        "susp"_a = false
    );

    // Explicit factories
    exp_m.def("ddt", &dsl::exp::ddt<scalar>);
    exp_m.def("ddt", &dsl::exp::ddt<Vec3>);
    exp_m.def(
        "div",
        [](const ScalarSurfField& flux, const ScalarVolField& phi)
        { return dsl::exp::div(flux, phi); }
    );
    exp_m.def("div", [](const ScalarSurfField& flux) { return dsl::exp::div(flux); });

    // Direct flux divergence evaluation: div(phi) → VolumeField<scalar>
    // Computes (1/V) * sum(phi_f) per cell — Gauss divergence theorem on face flux.
    exp_m.def(
        "div_flux",
        [](const ScalarSurfField& flux)
        {
            const auto& mesh = flux.mesh();
            auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
            fvcc::VolumeField<NeoN::scalar> result(flux.exec(), "divFlux", mesh, bcs);

            const auto owner = mesh.faceOwner().view();
            const auto neighbour = mesh.faceNeighbour().view();
            const auto phiView = flux.internalVector().view();
            const auto vol = mesh.cellVolumes().view();
            auto outView = result.internalVector().view();
            const auto nInternalFaces = mesh.nInternalFaces();
            const auto nCells = mesh.nCells();

            // Zero out
            NeoN::fill(result.internalVector(), 0.0);

            // Internal faces: owner gets +phi, neighbour gets -phi
            NeoN::parallelFor(
                flux.exec(),
                {0, nInternalFaces},
                NEON_LAMBDA(const NeoN::localIdx facei) {
                    Kokkos::atomic_add(&outView[owner[facei]], phiView[facei]);
                    Kokkos::atomic_add(&outView[neighbour[facei]], -phiView[facei]);
                },
                "divFlux_internal"
            );

            // Boundary faces: owner gets +phi
            NeoN::parallelFor(
                flux.exec(),
                {nInternalFaces, static_cast<NeoN::localIdx>(flux.internalVector().size())},
                NEON_LAMBDA(const NeoN::localIdx facei) {
                    Kokkos::atomic_add(&outView[owner[facei]], phiView[facei]);
                },
                "divFlux_boundary"
            );

            // Divide by cell volume
            NeoN::parallelFor(
                flux.exec(),
                {0, nCells},
                NEON_LAMBDA(const NeoN::localIdx celli) { outView[celli] /= vol[celli]; },
                "divFlux_normalize"
            );

            return result;
        },
        "flux"_a,
        "Compute divergence of face flux: (1/V)*sum(phi_f) per cell"
    );
    exp_m.def("laplacian", &dsl::exp::laplacian<scalar>);
    exp_m.def("laplacian", &dsl::exp::laplacian<Vec3>);
    exp_m.def(
        "grad",
        [](const ScalarVolField& phi) { return dsl::exp::grad(phi); },
        "Gradient of a scalar field (returns Vec3)"
    );
    exp_m.def(
        "grad",
        [](const VectorVolField& phi) { return dsl::exp::grad(phi); },
        "Gradient of a Vec3 field (returns Tensor)"
    );
    exp_m.def("source", &dsl::exp::source<scalar>);
    exp_m.def("source", &dsl::exp::source<Vec3>);

    // Direct gradient evaluation — returns VolumeField (not lazy SpatialOperator).
    // Uses GaussGreen gradient with linear interpolation.
    exp_m.def(
        "grad_field",
        [](const ScalarVolField& phi)
        {
            fvcc::GaussGreenGrad gradOp(phi.exec(), phi.mesh());
            return gradOp.grad(phi, dsl::Coeff(1.0));
        },
        "phi"_a,
        "Compute gradient of scalar field, returning VolumeField<Vec3>"
    );

    exp_m.def(
        "grad_field",
        [](const VectorVolField& phi)
        {
            fvcc::GaussGreenGradVec3 gradOp(phi.exec(), phi.mesh());
            return gradOp.grad(phi, dsl::Coeff(1.0));
        },
        "phi"_a,
        "Compute gradient of vector field, returning VolumeField<Tensor>"
    );

    using TensorVolField = NeoN::finiteVolume::cellCentred::VolumeField<Tensor>;

    exp_m.def(
        "div_tensor",
        [](const TensorVolField& T)
        {
            fvcc::GaussGreenDivTensor divOp(T.exec(), T.mesh());
            return divOp.div(T, dsl::Coeff(1.0));
        },
        "T"_a,
        "Gauss divergence of a tensor field: (1/V) * sum(Sf & Tf)"
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
}

} // namespace NeoN::bindings
