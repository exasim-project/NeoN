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

TEMPLATE_TEST_CASE("DivOperator", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);

    // compute corresponding uniform faceFlux
    // TODO this should be handled outside of the unit test
    fvcc::SurfaceField<scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    fill(faceFlux.internalVector(), 1.0);
    fill(faceFlux.boundaryData().value(), 1.0);
    // left boundary face (patch 0) has opposite orientation to the flow
    auto bFaceFlux = faceFlux.boundaryData().value().view();
    parallelFor(
        exec, {0, 1}, NEON_LAMBDA(const localIdx i) { bFaceFlux[i] = -1.0; }
    );

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    fill(phi.internalVector(), one<TestType>());
    fill(phi.boundaryData().value(), one<TestType>());
    phi.correctBoundaryConditions();

    auto result = Vector<TestType>(exec, phi.size());
    fill(result, zero<TestType>());

    SECTION("Construct from Token" + execName)
    {
        Input input = TokenList({std::string("Gauss"), std::string("linear")});
        fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
    }

    SECTION("Construct from Dictionary" + execName)
    {
        Input input = Dictionary(
            {{std::string("DivOperator"), std::string("Gauss")},
             {std::string("surfaceInterpolation"), std::string("linear")}}
        );
        auto op = fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
        op.explicitOperation(result);

        // divergence of a uniform field should be zero
        auto outHost = result.copyToHost();
        auto outHostView = outHost.view();
        for (int i = 0; i < result.size(); i++)
        {
            REQUIRE(outHostView[i] == zero<TestType>());
        }
    }

    SECTION("Implicit operation " + execName)
    {
        if constexpr (std::is_same_v<TestType, scalar>)
        {
            Input input = Dictionary(
                {{std::string("DivOperator"), std::string("Gauss")},
                 {std::string("surfaceInterpolation"), std::string("linear")}}
            );
            auto op = fvcc::DivOperator(Operator::Type::Implicit, faceFlux, phi, input);
            auto ls = la::createEmptyLinearSystem<TestType>(mesh);
            op.implicitOperation(ls);

            // the divergence of a uniform field under a conservative flux is zero,
            // so A*phi - b must vanish for every cell
            auto res = Vector<scalar>(exec, mesh.nCells(), 0.0);
            computeResidual(ls.matrix(), ls.rhs(), phi.internalVector(), res);

            auto resExp = std::vector<NeoN::scalar>(res.size(), 0);
            REQUIRE_THAT(res, Equals(resExp, Approx {1e-12}));
        }
    }
}

TEST_CASE("DivOperator implicit boundary contributions are accumulated")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);

    fvcc::SurfaceField<NeoN::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    fill(faceFlux.internalVector(), 1.0);
    fill(faceFlux.boundaryData().value(), 1.0);

    Input input = TokenList({std::string("Gauss"), std::string("linear")});

    SECTION("bRhs accumulates for fixedValue BC on " + execName)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs;
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(
            mesh,
            Dictionary({{"type", std::string("fixedValue")}, {"fixedValue", NeoN::scalar(1.0)}}),
            0
        ));
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(
            mesh,
            Dictionary({{"type", std::string("fixedValue")}, {"fixedValue", NeoN::scalar(2.0)}}),
            1
        ));
        auto phi = fvcc::VolumeField<NeoN::scalar>(exec, "phi", mesh, bcs);
        fill(phi.internalVector(), NeoN::scalar(1.0));
        phi.correctBoundaryConditions();

        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh);
        dsl::SpatialOperator divOp = dsl::imp::div(faceFlux, phi);
        divOp.read(input);

        divOp.implicitOperation(ls);
        auto bRhsFirst = ls.boundaryRhs().copyToHost();

        // second call without reset: bRhs must accumulate, not overwrite
        divOp.implicitOperation(ls);
        auto bRhsSecond = ls.boundaryRhs().copyToHost();

        auto bRhsFirstV = bRhsFirst.view();
        auto bRhsSecondV = bRhsSecond.view();
        for (localIdx i = 0; i < bRhsFirstV.size(); i++)
        {
            REQUIRE(NeoN::mag(bRhsFirstV[i]) > 0);
            REQUIRE(bRhsSecondV[i] == Catch::Approx(2.0 * bRhsFirstV[i]).margin(1e-8));
        }
    }

    SECTION("bValues has correct sign and accumulates for fixedGradient BC on " + execName)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs;
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(
            mesh,
            Dictionary(
                {{"type", std::string("fixedGradient")}, {"fixedGradient", NeoN::scalar(1.0)}}
            ),
            0
        ));
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(
            mesh,
            Dictionary(
                {{"type", std::string("fixedGradient")}, {"fixedGradient", NeoN::scalar(1.0)}}
            ),
            1
        ));
        auto phi = fvcc::VolumeField<NeoN::scalar>(exec, "phi", mesh, bcs);
        fill(phi.internalVector(), NeoN::scalar(1.0));
        phi.correctBoundaryConditions();

        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh);
        dsl::SpatialOperator divOp = dsl::imp::div(faceFlux, phi);
        divOp.read(input);

        divOp.implicitOperation(ls);
        auto bValuesFirst = ls.boundaryMatrix().values().copyToHost();
        auto bRhsFirst = ls.boundaryRhs().copyToHost();

        // second call without reset: contributions must accumulate, not overwrite
        divOp.implicitOperation(ls);
        auto bValuesSecond = ls.boundaryMatrix().values().copyToHost();
        auto bRhsSecond = ls.boundaryRhs().copyToHost();

        auto bValuesFirstV = bValuesFirst.view();
        auto bValuesSecondV = bValuesSecond.view();
        auto bRhsFirstV = bRhsFirst.view();
        auto bRhsSecondV = bRhsSecond.view();
        for (localIdx i = 0; i < bValuesFirstV.size(); i++)
        {
            // bValues stores the inverse of the diagonal contribution (bValues -= valueMat)
            // so bValues must be negative when the diagonal contribution is positive
            REQUIRE(bValuesFirstV[i] < 0);
            REQUIRE(bValuesSecondV[i] == Catch::Approx(2.0 * bValuesFirstV[i]).margin(1e-8));
        }
        for (localIdx i = 0; i < bRhsFirstV.size(); i++)
        {
            REQUIRE(bRhsSecondV[i] == Catch::Approx(2.0 * bRhsFirstV[i]).margin(1e-8));
        }
    }
}

TEMPLATE_TEST_CASE(
    "Face based and cellbased iteration give same results", "[template]", NeoN::scalar
)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);

    fvcc::SurfaceField<NeoN::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    fill(faceFlux.internalVector(), 1.0);
    fill(faceFlux.boundaryData().value(), 1.0);

    Input input = TokenList({std::string("Gauss"), std::string("linear")});

    auto lsFaceBased = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh);

    auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
    auto lsCellBased = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh, cellIterator);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    dsl::SpatialOperator divOp = dsl::imp::div(faceFlux, phi);
    divOp.read(input);
    divOp.implicitOperation(lsFaceBased);
    divOp.implicitOperation(lsCellBased);

    REQUIRE_THAT(
        lsFaceBased.matrix().values(), Equals(lsCellBased.matrix().values(), Approx {1e-12})
    );
}

} // namespace NeoN
