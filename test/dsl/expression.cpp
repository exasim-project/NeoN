// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "common.hpp"

namespace dsl = NeoN::dsl;


// TEST_CASE("Expression")
TEMPLATE_TEST_CASE("Expression", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = NeoN::createSingleCellMesh(exec);
    auto sp = NeoN::finiteVolume::cellCentred::SparsityPattern {mesh};

    const size_t size {1};
    NeoN::BoundaryData<TestType> bf(exec, mesh.boundaryMesh().offset());

    std::vector<fvcc::VolumeBoundary<TestType>> bcs {};
    NeoN::Field<TestType> fA(exec, 1, 2.0 * NeoN::one<TestType>());
    NeoN::Field<NeoN::scalar> scaleField(exec, 1, 4.0);
    auto vf = fvcc::VolumeField(exec, "vf", mesh, fA, bf, bcs);


    SECTION("Create equation and perform explicit Operation on " + execName)
    {
        // TODO conversion from Dummy to SpatialOperator is not automatic
        dsl::SpatialOperator<TestType> a = Dummy<TestType>(vf);
        dsl::SpatialOperator<TestType> b = Dummy<TestType>(vf);

        auto eqnA = a + b;
        auto eqnB = scaleField * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf))
                  + 2 * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf));
        auto eqnC = dsl::Expression<TestType>(2 * a - b);
        auto eqnD = 3 * (2 * a - b);
        auto eqnE = (2 * a - b) + (2 * a - b);
        auto eqnF = (2 * a - b) - (2 * a - b);

        REQUIRE(eqnA.size() == 2);
        REQUIRE(eqnB.size() == 2);
        REQUIRE(eqnC.size() == 2);

        // 2 + 2 = 4
        REQUIRE(getField(eqnA.explicitOperation(size)) == 4 * NeoN::one<TestType>());
        // 4*2 + 2*2 = 12
        REQUIRE(getField(eqnB.explicitOperation(size)) == 12 * NeoN::one<TestType>());
        // 2*2 - 2 = 2
        REQUIRE(getField(eqnC.explicitOperation(size)) == 2 * NeoN::one<TestType>());
        // 3*(2*2 - 2) = 6
        REQUIRE(getField(eqnD.explicitOperation(size)) == 6 * NeoN::one<TestType>());
        // 2*2 - 2 + 2*2 - 2 = 4
        REQUIRE(getField(eqnE.explicitOperation(size)) == 4 * NeoN::one<TestType>());
        // 2*2 - 2 - 2*2 + 2 = 0
        REQUIRE(getField(eqnF.explicitOperation(size)) == 0 * NeoN::one<TestType>());
    }

    auto ls = NeoN::la::createEmptyLinearSystem<
        TestType,
        NeoN::localIdx,
        NeoN::finiteVolume::cellCentred::SparsityPattern>(sp);

    SECTION("Create equation and perform implicit Operation on " + execName)
    {
        // TODO conversion from Dummy to SpatialOperator is not automatic
        dsl::SpatialOperator<TestType> a = Dummy<TestType>(vf, Operator::Type::Implicit);
        dsl::SpatialOperator<TestType> b = Dummy<TestType>(vf, Operator::Type::Implicit);

        auto eqnA = a + b;
        auto eqnB =
            scaleField
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
