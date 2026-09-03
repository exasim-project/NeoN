// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

// fixedFluxPressure differs from fixedGradient in exactly one respect: refGrad is owned by an
// external caller (NeoFOAM::constrainPressure sets it once per pressure corrector), so
// correctBoundaryCondition must READ it and never overwrite it with a stored uniform. These
// tests pin that contract: a per-face refGrad survives repeated correction, and the value is
// re-derived as owner + refGrad/deltaCoeffs each time (including after refGrad or the internal
// field is moved by the caller).
TEST_CASE("fixedFluxPressure")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // unit cube mesh: one cell, six single-face boundary patches, deltaCoeffs == 2.0
    auto mesh = NeoN::createSingleCellMesh(exec);
    const NeoN::scalar deltaCoeff = 2.0;
    const auto nBnd = mesh.nBoundaryFaces();

    NeoN::Field<NeoN::scalar> domainVector(exec, mesh.nCells(), mesh.boundaryMesh().offset());
    NeoN::fill(domainVector.internalVector(), 1.0);
    NeoN::fill(domainVector.boundaryData().refGrad(), 0.0);
    NeoN::fill(domainVector.boundaryData().refValue(), -1.0);
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
                "fixedFluxPressure", mesh, dict, patchID
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

    SECTION("zero refGrad behaves as zeroGradient" + execName)
    {
        correctAll();

        auto values = domainVector.boundaryData().value().copyToHost();
        auto valuesView = values.view();
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            REQUIRE(valuesView[i] == 1.0);
        }
    }

    SECTION("per-face refGrad is preserved and drives the value" + execName)
    {
        // A different refGrad on every boundary face: a BC that (like fixedGradient) wrote a
        // stored uniform back into refGrad would flatten this and fail below.
        std::vector<NeoN::scalar> seeded(static_cast<std::size_t>(nBnd));
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            seeded[static_cast<std::size_t>(i)] = 10.0 + static_cast<NeoN::scalar>(i);
        }
        domainVector.boundaryData().refGrad() = NeoN::Vector<NeoN::scalar>(exec, seeded);

        // Correct repeatedly: the BC must be idempotent and must not clobber refGrad.
        for (int pass = 0; pass < 3; ++pass)
        {
            correctAll();

            auto refGrad = domainVector.boundaryData().refGrad().copyToHost();
            auto values = domainVector.boundaryData().value().copyToHost();
            auto fractions = domainVector.boundaryData().valueFraction().copyToHost();
            auto refGradView = refGrad.view();
            auto valuesView = values.view();
            auto fractionsView = fractions.view();

            for (NeoN::localIdx i = 0; i < nBnd; ++i)
            {
                const auto expected = seeded[static_cast<std::size_t>(i)];
                REQUIRE(refGradView[i] == expected);
                REQUIRE_THAT(
                    valuesView[i], Catch::Matchers::WithinAbs(1.0 + expected / deltaCoeff, 1e-12)
                );
                // gradient-only: the mixed-BC blend must select refGrad exclusively
                REQUIRE(fractionsView[i] == 0.0);
            }
        }

        // A moved internal field re-derives the value from the SAME refGrad.
        NeoN::fill(domainVector.internalVector(), 5.0);
        correctAll();

        auto refGrad = domainVector.boundaryData().refGrad().copyToHost();
        auto values = domainVector.boundaryData().value().copyToHost();
        auto refGradView = refGrad.view();
        auto valuesView = values.view();
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            const auto expected = seeded[static_cast<std::size_t>(i)];
            REQUIRE(refGradView[i] == expected);
            REQUIRE_THAT(
                valuesView[i], Catch::Matchers::WithinAbs(5.0 + expected / deltaCoeff, 1e-12)
            );
        }
    }

    SECTION("an externally updated refGrad is picked up" + execName)
    {
        // The constrainPressure use case: the caller rewrites refGrad each corrector and the
        // BC must track it rather than a value captured at construction.
        NeoN::fill(domainVector.boundaryData().refGrad(), 4.0);
        correctAll();
        {
            auto values = domainVector.boundaryData().value().copyToHost();
            auto valuesView = values.view();
            for (NeoN::localIdx i = 0; i < nBnd; ++i)
            {
                REQUIRE_THAT(valuesView[i], Catch::Matchers::WithinAbs(3.0, 1e-12));
            }
        }

        NeoN::fill(domainVector.boundaryData().refGrad(), -4.0);
        correctAll();
        {
            auto values = domainVector.boundaryData().value().copyToHost();
            auto valuesView = values.view();
            for (NeoN::localIdx i = 0; i < nBnd; ++i)
            {
                REQUIRE_THAT(valuesView[i], Catch::Matchers::WithinAbs(-1.0, 1e-12));
            }
        }
    }
}
