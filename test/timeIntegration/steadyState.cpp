// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "../dsl/common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

using VolumeField = fvcc::VolumeField<NeoN::scalar>;

// Force instantiation so the static registration runs on MSVC
template class NeoN::timeIntegration::SteadyState<VolumeField>;

TEST_CASE("TimeIntegration - SteadyState")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Database db;
    NeoN::Dictionary timeIntegrationDict;
    timeIntegrationDict.insert("type", std::string("steadyState"));
    NeoN::Dictionary fvSolution;

    auto mesh = NeoN::createSingleCellMesh(exec);
    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "fieldCollection");
    fvcc::VolumeField<NeoN::scalar>& vf =
        fieldCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
            CreateVector {.name = "vf", .mesh = mesh, .value = 42.0, .timeIndex = 1}
        );

    SECTION("solve is a no-op on " + execName)
    {
        NeoN::timeIntegration::TimeIntegration<VolumeField> timeIntegrator(
            timeIntegrationDict, fvSolution
        );

        // Build a trivial expression (no operators) — solve must leave vf unchanged
        NeoN::dsl::Expression<NeoN::scalar> eqn(exec);

        auto before = vf.internalVector().copyToHost();
        timeIntegrator.solve(eqn, vf, 0.0, 1.0);
        auto after = vf.internalVector().copyToHost();

        REQUIRE(after.view()[0] == Catch::Approx(before.view()[0]));
    }

    SECTION("explicitIntegration returns false on " + execName)
    {
        NeoN::timeIntegration::TimeIntegration<VolumeField> timeIntegrator(
            timeIntegrationDict, fvSolution
        );
        REQUIRE_FALSE(timeIntegrator.explicitIntegration());
    }

    SECTION("registered under name 'steadyState' on " + execName)
    {
        // Constructing via the factory with type "steadyState" must not throw
        REQUIRE_NOTHROW(
            NeoN::timeIntegration::TimeIntegration<VolumeField>(timeIntegrationDict, fvSolution)
        );
    }
}
