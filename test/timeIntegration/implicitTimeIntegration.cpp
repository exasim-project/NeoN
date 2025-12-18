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

TEST_CASE("TimeIntegration")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Database db;
    auto mesh = NeoN::createSingleCellMesh(exec);
    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "fieldCollection");

    NeoN::Dictionary fvSchemes;
    NeoN::Dictionary ddtSchemes;
    ddtSchemes.insert("type", std::string("forwardEuler"));
    ddtSchemes.insert("ddt(vf)", std::string("BDF1"));
    fvSchemes.insert("ddtSchemes", ddtSchemes);
    NeoN::Dictionary fvSolution;

    fvcc::VolumeField<NeoN::scalar>& vf =
        fieldCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
            CreateVector {.name = "vf", .mesh = mesh, .value = 2.0, .timeIndex = 1}
        );

    SECTION("Create expression and perform explicitOperation on " + execName)
    {
        auto dummy = Dummy(vf);
        NeoN::dsl::TemporalOperator ddtOperator = NeoN::dsl::imp::ddt(vf);

        // ddt(U) = f
        auto eqn = ddtOperator + dummy;
        double dt {2.0};
        double time {1.0};


        // int(ddt(U)) + f = 0
        // (U^1-U^0)/dt = -f
        // U^1 = - f * dt + U^0, where dt = 2, f=1, U^0=2.0 -> U^1=-2.0
        NeoN::dsl::solve(eqn, vf, time, dt, fvSchemes, fvSolution);
        REQUIRE(getVector(vf.internalVector()) == -2.0);
    }
}
