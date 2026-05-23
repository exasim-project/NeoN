// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


namespace fvcc = NeoN::finiteVolume::cellCentred;

using Operator = NeoN::dsl::Operator;

namespace NeoN
{

auto makeFvSchemes = []()
{
    NeoN::Dictionary fvSchemes;
    NeoN::Dictionary divSchemes;
    divSchemes.insert(
        "div(faceFlux,phi)", NeoN::TokenList({std::string("Gauss"), std::string("linear")})
    );
    fvSchemes.insert("divSchemes", divSchemes);

    NeoN::Dictionary lapSchemes;
    lapSchemes.insert(
        "laplacian(gamma,phi)",
        NeoN::TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")})
    );
    fvSchemes.insert("laplacianSchemes", lapSchemes);
    return fvSchemes;
};

TEMPLATE_TEST_CASE("Div + Laplacian Operator ", "[template]", NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);

    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volBCs);
    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());
    NeoN::fill(phi.boundaryData().value(), NeoN::zero<TestType>());
    phi.correctBoundaryConditions();

    auto fvSchemes = makeFvSchemes();

    fvcc::SurfaceField<NeoN::scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    NeoN::fill(faceFlux.internalVector(), 1.0);

    fvcc::SurfaceField<NeoN::scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    NeoN::fill(gamma.internalVector(), 1.0);

    auto expr = NeoN::dsl::imp::div(faceFlux, phi) + NeoN::dsl::imp::laplacian(gamma, phi);

    expr.read(fvSchemes);

    auto t = NeoN::scalar(1.0);
    auto dt = NeoN::scalar(1.0);

    SECTION("Can assemble to scalar matrix with rhs<Vec3>")
    {
        auto ls = expr.template assemble<NeoN::scalar>(mesh, t, dt);

        // Matrix coefficients are scalar; rhs values are Vec3.
        static_assert(std::is_same_v<
                      std::decay_t<decltype(ls.matrix().values())>,
                      NeoN::Vector<NeoN::scalar>>);
        static_assert(std::is_same_v<std::decay_t<decltype(ls.rhs())>, NeoN::Vector<NeoN::Vec3>>);

        // Assembly must have produced non-zero matrix entries.
        auto matHost = ls.matrix().values().copyToHost();
        auto matView = matHost.view();
        NeoN::scalar matAbsSum = 0.0;
        for (localIdx i = 0; i < matView.size(); ++i)
        {
            matAbsSum += std::abs(matView[i]);
        }
        REQUIRE(matAbsSum > 0.0);
    }

    SECTION("Can assemble to Vec3 matrix with rhs<vec3>")
    {
        auto ls = expr.template assemble<NeoN::Vec3>(mesh, t, dt);

        // Matrix coefficients and rhs values are both Vec3.
        static_assert(std::is_same_v<
                      std::decay_t<decltype(ls.matrix().values())>,
                      NeoN::Vector<NeoN::Vec3>>);
        static_assert(std::is_same_v<std::decay_t<decltype(ls.rhs())>, NeoN::Vector<NeoN::Vec3>>);

        // Assembly must have produced non-zero matrix entries.
        auto matHost = ls.matrix().values().copyToHost();
        auto matView = matHost.view();
        NeoN::scalar matMagSum = 0.0;
        for (localIdx i = 0; i < matView.size(); ++i)
        {
            matMagSum += mag(matView[i]);
        }
        REQUIRE(matMagSum > 0.0);
    }

    SECTION("Can solve with multiple RHS")
    {
        auto ls = expr.template assemble<NeoN::scalar>(mesh, t, dt);

        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
        };

        auto solver = NeoN::la::Solver(exec, solverDict);
        Vector<Vec3> x(exec, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}});
        auto solverStats = solver.solve(ls, x);

        REQUIRE(solverStats.entries.size() > 0);
        const auto& stats = solverStats.entries[0];

        // The solver must have iterated at least once.
        REQUIRE(stats.numIter > 0);

        // Residual should not have grown.
        REQUIRE(stats.finalResNorm <= stats.initResNorm);

        // x should be non-zero after solving.
        auto xHost = x.copyToHost();
        auto xView = xHost.view();
        NeoN::scalar xMagSum = 0.0;
        for (localIdx i = 0; i < xView.size(); ++i)
        {
            xMagSum += mag(xView[i]);
        }
        REQUIRE(xMagSum > 0.0);
    }
}

} // namespace NeoN
