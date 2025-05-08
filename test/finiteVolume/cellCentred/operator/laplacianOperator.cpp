// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


using NeoN::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoN::finiteVolume::cellCentred::VolumeField;
using NeoN::finiteVolume::cellCentred::SurfaceField;

namespace NeoN
{

template<typename T>
using I = std::initializer_list<T>;

TEMPLATE_TEST_CASE("laplacianOperator fixedValue", "[template]", scalar, Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const NeoN::localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);
    auto sp = la::SparsityPattern {mesh};

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 2.0);

    auto [boundaryType, firstValue, lastValue] = GENERATE(
        std::tuple<std::string, scalar, scalar> {"fixedValue", 0.5, 10.5},
        std::tuple<std::string, scalar, scalar> {"fixedGradient", -10.0, 10}
    );

    SECTION(boundaryType)
    {
        std::vector<fvcc::VolumeBoundary<TestType>> bcs;
        bcs.push_back(fvcc::VolumeBoundary<TestType>(
            mesh,
            Dictionary(
                {{"type", std::string(boundaryType)}, {boundaryType, firstValue * one<TestType>()}}
            ),
            0
        ));
        bcs.push_back(fvcc::VolumeBoundary<TestType>(
            mesh,
            Dictionary(
                {{"type", std::string(boundaryType)}, {boundaryType, lastValue * one<TestType>()}}
            ),
            1
        ));

        auto phi = fvcc::VolumeField<TestType>(exec, "phi", mesh, bcs);
        parallelFor(
            phi.internalVector(),
            KOKKOS_LAMBDA(const localIdx i) { return scalar(i + 1) * one<TestType>(); }
        );
        phi.correctBoundaryConditions();

        Input input =
            TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")});

        SECTION("Construct from Token" + execName)
        {
            fvcc::LaplacianOperator<TestType> lapOp(
                dsl::Operator::Type::Implicit, gamma, phi, input
            );
        }

        SECTION("explicit laplacian operator for constant field on " + execName)
        {
            dsl::SpatialOperator lapOp = dsl::exp::laplacian(gamma, phi);
            lapOp.read(input);
            Vector<TestType> source(exec, nCells, zero<TestType>());
            lapOp.explicitOperation(source);
            auto sourceHost = source.copyToHost();
            auto sourceV = sourceHost.view();
            for (NeoN::localIdx i = 0; i < nCells; i++)
            {
                // the laplacian of a linear function is 0
                REQUIRE(mag(sourceV[i]) == Catch::Approx(0.0).margin(1e-8));
            }
        }

        auto ls = la::createEmptyLinearSystem<TestType, localIdx>(mesh, sp);

        SECTION("implicit laplacian operator of constant field on " + execName)
        {
            dsl::SpatialOperator lapOp = dsl::imp::laplacian(gamma, phi);
            lapOp.read(input);
            // currently only defined for scalar types
            if constexpr (std::is_same_v<TestType, scalar>)
            {
                lapOp.implicitOperation(ls);
                auto res = Vector<scalar>(phi.internalVector());
                fill(res, 1.0);

                computeResidual(ls.matrix(), ls.rhs(), phi.internalVector(), res);

                auto resHost = res.copyToHost();
                auto resV = resHost.view();
                for (localIdx celli = 0; celli < resV.size(); celli++)
                {
                    // the laplacian of a linear function is 0
                    REQUIRE(resV[celli] == Catch::Approx(0.0).margin(1e-8));
                }
            }
        }

        SECTION("implicit laplacian operator scale" + execName)
        {
            if constexpr (std::is_same_v<TestType, scalar>)
            {
                ls.reset();
                dsl::SpatialOperator lapOp = dsl::imp::laplacian(gamma, phi);
                lapOp.read(input);
                lapOp = dsl::Coeff(-0.5) * lapOp;

                lapOp.implicitOperation(ls);

                auto res = Vector<scalar>(phi.internalVector());
                computeResidual(ls.matrix(), ls.rhs(), phi.internalVector(), res);

                auto resHost = res.copyToHost();
                auto resV = resHost.view();
                for (localIdx celli = 0; celli < resV.size(); celli++)
                {
                    // the laplacian of a linear function is 0
                    REQUIRE(resV[celli] == Catch::Approx(0.0).margin(1e-8));
                }
            }
        }
    }
}

} // namespace NeoN
