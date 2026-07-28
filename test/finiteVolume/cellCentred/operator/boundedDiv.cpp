// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

#include <algorithm>


namespace fvcc = NeoN::finiteVolume::cellCentred;

using Operator = NeoN::dsl::Operator;

namespace NeoN
{

TEST_CASE("BoundedDiv is registered in the div operator factory")
{
    auto entries = fvcc::DivOperatorFactory<scalar>::entries();
    REQUIRE(std::find(entries.begin(), entries.end(), std::string("bounded")) != entries.end());
}

// The bounded correction is designed around two exact invariants:
//   1. it vanishes identically for a spatially uniform field, whatever the flux looks like
//      (the whole point of the scheme: a uniform field can never be pushed off by continuity
//      noise), and
//   2. it vanishes identically for a divergence-free flux, whatever the field looks like (no
//      continuity error to correct for). These two SECTIONs check each invariant directly
//      instead of hand-deriving expected matrix/residual values.
TEMPLATE_TEST_CASE("BoundedDiv", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);

    auto boundedUpwind = []() {
        return Input(TokenList({std::string("bounded"), std::string("Gauss"), std::string("upwind")}
        ));
    };
    auto plainUpwind = []() {
        return Input(TokenList({std::string("Gauss"), std::string("upwind")}));
    };

    SECTION("a uniform field is unaffected by continuity error on " + execName)
    {
        // Non-conservative flux: internal and boundary values are independently randomized, so
        // the discrete divergence is generically nonzero in every cell.
        fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
        Catch::randomizeVector(faceFlux.internalVector());
        Catch::randomizeVector(faceFlux.boundaryData().value());

        fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
        fill(phi.internalVector(), one<TestType>());
        fill(phi.boundaryData().value(), one<TestType>());
        phi.correctBoundaryConditions();

        auto result = Vector<TestType>(exec, phi.size());
        fill(result, zero<TestType>());
        fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, boundedUpwind())
            .explicitOperation(result);

        auto resHost = result.copyToHost();
        auto resHostV = resHost.view();
        for (localIdx i = 0; i < resHostV.size(); i++)
        {
            REQUIRE(mag(resHostV[i]) < 1e-10);
        }

        if constexpr (std::is_same_v<TestType, scalar>)
        {
            auto ls = la::createEmptyLinearSystem<TestType>(mesh);
            fvcc::DivOperator(Operator::Type::Implicit, faceFlux, phi, boundedUpwind())
                .implicitOperation(ls);

            auto res = Vector<scalar>(exec, mesh.nCells(), 0.0);
            computeResidual(ls.matrix(), ls.rhs(), phi.internalVector(), res);

            auto resExp = std::vector<NeoN::scalar>(static_cast<size_t>(res.size()), 0);
            REQUIRE_THAT(res, Equals(resExp, Approx {1e-10}));
        }
    }

    SECTION("reduces to the inner scheme for a divergence-free flux on " + execName)
    {
        // Conservative flux: uniform magnitude, correctly oriented at the domain boundaries so
        // every cell's net outflow is exactly zero.
        fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
        fill(faceFlux.internalVector(), 1.0);
        fill(faceFlux.boundaryData().value(), 1.0);
        auto bFaceFlux = faceFlux.boundaryData().value().view();
        parallelFor(
            exec, {0, 1}, NEON_LAMBDA(const localIdx i) { bFaceFlux[i] = -1.0; }
        );

        fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
        Catch::randomizeVector(phi);
        phi.correctBoundaryConditions();

        auto resultBounded = Vector<TestType>(exec, phi.size());
        auto resultPlain = Vector<TestType>(exec, phi.size());
        fill(resultBounded, zero<TestType>());
        fill(resultPlain, zero<TestType>());

        fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, boundedUpwind())
            .explicitOperation(resultBounded);
        fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, plainUpwind())
            .explicitOperation(resultPlain);

        REQUIRE_THAT(resultBounded, Equals(resultPlain, Approx {1e-10}));

        if constexpr (std::is_same_v<TestType, scalar>)
        {
            auto lsBounded = la::createEmptyLinearSystem<TestType>(mesh);
            auto lsPlain = la::createEmptyLinearSystem<TestType>(mesh);
            fvcc::DivOperator(Operator::Type::Implicit, faceFlux, phi, boundedUpwind())
                .implicitOperation(lsBounded);
            fvcc::DivOperator(Operator::Type::Implicit, faceFlux, phi, plainUpwind())
                .implicitOperation(lsPlain);

            REQUIRE_THAT(
                lsBounded.matrix().values(), Equals(lsPlain.matrix().values(), Approx {1e-10})
            );
        }
    }
}

} // namespace NeoN
