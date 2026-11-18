// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

// fixedEnergy differs from fixedValue in exactly one respect: refValue is owned by an
// external caller (mapEnergyBoundaryFromTemperature sets it once per outer iteration from
// the current temperature boundary value), so correctBoundaryCondition must READ it and
// never overwrite it with a stored uniform. These tests pin that contract: a per-face
// refValue survives repeated correction, and the value is re-derived as exactly refValue
// each time (including after refValue or the internal field is moved by the caller).
TEST_CASE("fixedEnergy")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // unit cube mesh: one cell, six single-face boundary patches
    auto mesh = NeoN::createSingleCellMesh(exec);
    const auto nBnd = mesh.nBoundaryFaces();

    NeoN::Field<NeoN::scalar> domainVector(exec, mesh.nCells(), mesh.boundaryMesh().offset());
    NeoN::fill(domainVector.internalVector(), 1.0);
    NeoN::fill(domainVector.boundaryData().refGrad(), -1.0);
    NeoN::fill(domainVector.boundaryData().refValue(), 0.0);
    NeoN::fill(domainVector.boundaryData().valueFraction(), -1.0);
    NeoN::fill(domainVector.boundaryData().value(), -1.0);

    NeoN::Dictionary dict;
    std::vector<
        std::unique_ptr<NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>>>
        boundaries;
    for (NeoN::localIdx patchID = 0; patchID < mesh.nBoundaries(); ++patchID)
    {
        boundaries.push_back(
            NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                "fixedEnergy", mesh, dict, patchID
            )
        );
    }
    auto correctAll = [&]()
    {
        for (auto& bc : boundaries)
        {
            bc->correctBoundaryCondition(domainVector);
        }
    };

    SECTION("zero refValue behaves as a zero fixedValue" + execName)
    {
        correctAll();

        auto values = domainVector.boundaryData().value().copyToHost();
        auto fractions = domainVector.boundaryData().valueFraction().copyToHost();
        auto gradients = domainVector.boundaryData().refGrad().copyToHost();
        auto valuesView = values.view();
        auto fractionsView = fractions.view();
        auto gradientsView = gradients.view();
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            REQUIRE(valuesView[i] == 0.0);
            // value-only: the mixed-BC blend must select refValue exclusively
            REQUIRE(fractionsView[i] == 1.0);
            REQUIRE(gradientsView[i] == 0.0);
        }
    }

    SECTION("per-face refValue is preserved and drives the value" + execName)
    {
        // A different refValue on every boundary face: a BC that (like fixedValue) wrote a
        // stored uniform back into refValue would flatten this and fail below.
        std::vector<NeoN::scalar> seeded(static_cast<sizeType>(nBnd));
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            seeded[static_cast<sizeType>(i)] = 1317.2 + static_cast<NeoN::scalar>(i);
        }
        domainVector.boundaryData().refValue() = NeoN::Vector<NeoN::scalar>(exec, seeded);

        // Correct repeatedly: the BC must be idempotent and must not clobber refValue.
        for (int pass = 0; pass < 3; ++pass)
        {
            correctAll();

            auto refValue = domainVector.boundaryData().refValue().copyToHost();
            auto values = domainVector.boundaryData().value().copyToHost();
            auto refValueView = refValue.view();
            auto valuesView = values.view();

            for (NeoN::localIdx i = 0; i < nBnd; ++i)
            {
                const auto expected = seeded[static_cast<sizeType>(i)];
                REQUIRE(refValueView[i] == expected);
                REQUIRE_THAT(valuesView[i], Catch::Matchers::WithinAbs(expected, 1e-12));
            }
        }

        // A moved internal field does not perturb the value -- fixedEnergy is purely
        // Dirichlet, unlike gradientEnergy's owner-dependent reconstruction.
        NeoN::fill(domainVector.internalVector(), 5.0);
        correctAll();

        auto values = domainVector.boundaryData().value().copyToHost();
        auto valuesView = values.view();
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            const auto expected = seeded[static_cast<sizeType>(i)];
            REQUIRE_THAT(valuesView[i], Catch::Matchers::WithinAbs(expected, 1e-12));
        }
    }

    SECTION("an externally updated refValue is picked up" + execName)
    {
        // The mapEnergyBoundaryFromTemperature use case: the caller rewrites refValue every
        // outer iteration and the BC must track it rather than a value captured at construction.
        NeoN::fill(domainVector.boundaryData().refValue(), 1317.2);
        correctAll();
        {
            auto values = domainVector.boundaryData().value().copyToHost();
            auto valuesView = values.view();
            for (NeoN::localIdx i = 0; i < nBnd; ++i)
            {
                REQUIRE_THAT(valuesView[i], Catch::Matchers::WithinAbs(1317.2, 1e-12));
            }
        }

        NeoN::fill(domainVector.boundaryData().refValue(), 1318.0);
        correctAll();
        {
            auto values = domainVector.boundaryData().value().copyToHost();
            auto valuesView = values.view();
            for (NeoN::localIdx i = 0; i < nBnd; ++i)
            {
                REQUIRE_THAT(valuesView[i], Catch::Matchers::WithinAbs(1318.0, 1e-12));
            }
        }
    }
}
