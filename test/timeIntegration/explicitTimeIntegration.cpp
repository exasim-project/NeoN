// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "../dsl/common.hpp"

#include "NeoN/NeoN.hpp"

// only needed for msvc
template class NeoN::timeIntegration::ForwardEuler<VolumeField>;

TEST_CASE("TimeIntegration: forwardEuler")
{
    // auto [execName, exec] = GENERATE(allAvailableExecutor());

    std::string execName = "SerialExecutor";
    NeoN::Executor exec = NeoN::SerialExecutor {};

    NeoN::Database db;
    auto mesh = NeoN::createSingleCellMesh(exec);
    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "fieldCollection");

    NeoN::Dictionary fvSchemes;
    NeoN::Dictionary timeIntegrationDict;
    timeIntegrationDict.insert("type", std::string("forwardEuler"));
    // ddtSchemes.insert("vf", std::string("BDF1"));
    fvSchemes.insert("timeIntegration", timeIntegrationDict);
    NeoN::Dictionary fvSolution;

    fvcc::VolumeField<NeoN::scalar>& vf =
        fieldCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
            CreateVector {.name = "vf", .mesh = mesh, .value = 2.0, .timeIndex = 1}
        );
    auto& vfOld = fvcc::oldTime(vf);
    vfOld.internalVector() = vf.internalVector();
    vfOld.correctBoundaryConditions();

    SECTION("Create expression and perform explicitOperation on " + execName)
    {
        auto dummy = Dummy(vf);
        auto ddtOperator = NeoN::dsl::ddt(vf);
        // ddt(U) = f
        NeoN::dsl::Expression<NeoN::scalar> eqn = ddtOperator + dummy;
        double dt {2.0};
        double time {1.0};

        // int(ddt(U)) + f = 0
        // (U^1-U^0)/dt = -f
        // U^1 = - f * dt + U^0, where dt = 2, f = 2, U^0=2.0 -> U^1=-2.0
        NeoN::dsl::solve(eqn, vf, time, dt, fvSchemes, fvSolution);
        NF_INFO("after solve");
        REQUIRE(getVector(vf.internalVector()) == -2.0);
    }
}
