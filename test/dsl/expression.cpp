// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "common.hpp"

namespace dsl = NeoN::dsl;


TEMPLATE_TEST_CASE("Expression", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = NeoN::createSingleCellMesh(exec);
    auto sp = NeoN::la::SparsityPattern {mesh};

    const size_t size {1};
    NeoN::BoundaryData<TestType> bf(exec, mesh.boundaryMesh().offset());

    std::vector<fvcc::VolumeBoundary<TestType>> bcs {};
    NeoN::Vector<TestType> fA(exec, 1, 2.0 * NeoN::one<TestType>());
    NeoN::Vector<NeoN::scalar> scaleVector(exec, 1, 4.0);
    auto vf = fvcc::VolumeField(exec, "vf", mesh, fA, bf, bcs);


    SECTION("Create equation and perform explicit Operation on " + execName)
    {
        // TODO conversion from Dummy to SpatialOperator is not automatic
        dsl::SpatialOperator<TestType> a = Dummy<TestType>(vf);
        dsl::SpatialOperator<TestType> b = Dummy<TestType>(vf);

        auto eqnA = a + b;
        auto eqnB = scaleVector * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf))
                  + 2 * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf));
        auto eqnC = dsl::Expression<TestType>(2 * a - b);
        auto eqnD = 3 * (2 * a - b);
        auto eqnE = (2 * a - b) + (2 * a - b);
        auto eqnF = (2 * a - b) - (2 * a - b);

        REQUIRE(eqnA.size() == 2);
        REQUIRE(eqnB.size() == 2);
        REQUIRE(eqnC.size() == 2);

        // 2 + 2 = 4
        REQUIRE(getVector(eqnA.explicitOperation(size)) == 4 * NeoN::one<TestType>());
        // 4*2 + 2*2 = 12
        REQUIRE(getVector(eqnB.explicitOperation(size)) == 12 * NeoN::one<TestType>());
        // 2*2 - 2 = 2
        REQUIRE(getVector(eqnC.explicitOperation(size)) == 2 * NeoN::one<TestType>());
        // 3*(2*2 - 2) = 6
        REQUIRE(getVector(eqnD.explicitOperation(size)) == 6 * NeoN::one<TestType>());
        // 2*2 - 2 + 2*2 - 2 = 4
        REQUIRE(getVector(eqnE.explicitOperation(size)) == 4 * NeoN::one<TestType>());
        // 2*2 - 2 - 2*2 + 2 = 0
        REQUIRE(getVector(eqnF.explicitOperation(size)) == 0 * NeoN::one<TestType>());
    }

    auto ls = NeoN::la::createEmptyLinearSystem<TestType, NeoN::localIdx>(mesh, sp);

    SECTION("Create equation and perform implicit Operation on " + execName)
    {
        // TODO conversion from Dummy to SpatialOperator is not automatic
        dsl::SpatialOperator<TestType> a = Dummy<TestType>(vf, Operator::Type::Implicit);
        dsl::SpatialOperator<TestType> b = Dummy<TestType>(vf, Operator::Type::Implicit);

        auto eqnA = a + b;
        auto eqnB =
            scaleVector
                * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf, Operator::Type::Implicit))
            + 2 * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf, Operator::Type::Implicit));
        auto eqnC = dsl::Expression<TestType>(2 * a - b);
        auto eqnD = 3 * (2 * a - b);
        auto eqnE = (2 * a - b) + (2 * a - b);
        auto eqnF = (2 * a - b) - (2 * a - b);

        REQUIRE(eqnA.size() == 2);
        REQUIRE(eqnB.size() == 2);
        REQUIRE(eqnC.size() == 2);

        // 2 + 2 = 4
        eqnA.implicitOperation(ls);
        REQUIRE(getDiag(ls) == 4 * NeoN::one<TestType>());
        REQUIRE(getRhs(ls) == 4 * NeoN::one<TestType>());

        // 4*2 + 2*2 = 12
        ls.reset();
        eqnB.implicitOperation(ls);
        REQUIRE(getDiag(ls) == 12 * NeoN::one<TestType>());
        REQUIRE(getRhs(ls) == 12 * NeoN::one<TestType>());

        // 2*2 - 2 = 2
        ls.reset();
        eqnC.implicitOperation(ls);
        REQUIRE(getDiag(ls) == 2 * NeoN::one<TestType>());
        REQUIRE(getRhs(ls) == 2 * NeoN::one<TestType>());

        // 3*(2*2 - 2) = 6
        ls.reset();
        eqnD.implicitOperation(ls);
        REQUIRE(getDiag(ls) == 6 * NeoN::one<TestType>());
        REQUIRE(getRhs(ls) == 6 * NeoN::one<TestType>());

        // 2*2 - 2 + 2*2 - 2 = 4
        ls.reset();
        eqnE.implicitOperation(ls);
        REQUIRE(getDiag(ls) == 4 * NeoN::one<TestType>());
        REQUIRE(getRhs(ls) == 4 * NeoN::one<TestType>());

        // // 2*2 - 2 - 2*2 + 2 = 0
        ls.reset();
        eqnF.implicitOperation(ls);
        REQUIRE(getDiag(ls) == 0 * NeoN::one<TestType>());
        REQUIRE(getRhs(ls) == 0 * NeoN::one<TestType>());
    }
}
