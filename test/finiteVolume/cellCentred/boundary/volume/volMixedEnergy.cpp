// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

// mixedEnergy is the energy counterpart of a mixed temperature patch (inletOutlet, freestream,
// ...). Like fixedEnergy it owns none of its data: refValue, refGrad and valueFraction are
// rewritten by mapEnergyBoundaryFromTemperature, and correctBoundaryCondition only evaluates
//   value = valueFraction*refValue + (1 - valueFraction)*(he_C + refGrad/delta).
// These tests pin that contract, in particular that the per-face valueFraction is honoured
// (gradientEnergy would force it to zero and lose the fixed inflow value).
TEST_CASE("mixedEnergy")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // unit cube mesh: one cell, six single-face boundary patches
    auto mesh = NeoN::createSingleCellMesh(exec);
    const auto nBnd = mesh.nBoundaryFaces();

    NeoN::Field<NeoN::scalar> domainVector(exec, mesh.nCells(), mesh.boundaryMesh().offset());
    const NeoN::scalar internal = 2.0;
    NeoN::fill(domainVector.internalVector(), internal);
    NeoN::fill(domainVector.boundaryData().refGrad(), 0.0);
    NeoN::fill(domainVector.boundaryData().refValue(), 0.0);
    NeoN::fill(domainVector.boundaryData().valueFraction(), 0.0);
    NeoN::fill(domainVector.boundaryData().value(), -1.0);

    auto deltaHost = mesh.boundaryMesh().deltaCoeffs().copyToHost();
    const auto delta = deltaHost.view();

    NeoN::Dictionary dict;
    std::vector<
        std::unique_ptr<NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>>>
        boundaries;
    for (NeoN::localIdx patchID = 0; patchID < mesh.nBoundaries(); ++patchID)
    {
        boundaries.push_back(
            NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                "mixedEnergy", mesh, dict, patchID
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
    auto setPerFace = [&](auto& boundaryVector, auto&& law)
    {
        std::vector<NeoN::scalar> seeded(static_cast<NeoN::sizeType>(nBnd));
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            seeded[static_cast<NeoN::sizeType>(i)] = law(i);
        }
        boundaryVector = NeoN::Vector<NeoN::scalar>(exec, seeded);
    };

    SECTION("valueFraction 1 behaves as a fixedValue on refValue" + execName)
    {
        NeoN::fill(domainVector.boundaryData().valueFraction(), 1.0);
        NeoN::fill(domainVector.boundaryData().refValue(), 1317.2);
        NeoN::fill(domainVector.boundaryData().refGrad(), 50.0); // must be ignored

        correctAll();

        auto values = domainVector.boundaryData().value().copyToHost();
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            REQUIRE_THAT(values.view()[i], Catch::Matchers::WithinAbs(1317.2, 1e-12));
        }
    }

    SECTION("valueFraction 0 behaves as a gradient patch" + execName)
    {
        NeoN::fill(domainVector.boundaryData().refValue(), 1317.2); // must be ignored
        NeoN::fill(domainVector.boundaryData().refGrad(), 0.0);

        correctAll();

        auto values = domainVector.boundaryData().value().copyToHost();
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            REQUIRE_THAT(values.view()[i], Catch::Matchers::WithinAbs(internal, 1e-12));
        }

        NeoN::fill(domainVector.boundaryData().refGrad(), 3.0);
        correctAll();

        values = domainVector.boundaryData().value().copyToHost();
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            REQUIRE_THAT(
                values.view()[i], Catch::Matchers::WithinAbs(internal + 3.0 / delta[i], 1e-12)
            );
        }
    }

    SECTION("a per-face valueFraction blends refValue and the gradient value" + execName)
    {
        // A different fraction, refValue and refGrad on every face: a BC that collapsed any of
        // them to one value (or forced the fraction to 0 or 1) would fail below.
        auto fraction = [](NeoN::localIdx i) { return 0.1 * static_cast<NeoN::scalar>(i); };
        auto refValue = [](NeoN::localIdx i) { return 1300.0 + static_cast<NeoN::scalar>(i); };
        auto refGrad = [](NeoN::localIdx i) { return -2.0 + static_cast<NeoN::scalar>(i); };
        setPerFace(domainVector.boundaryData().valueFraction(), fraction);
        setPerFace(domainVector.boundaryData().refValue(), refValue);
        setPerFace(domainVector.boundaryData().refGrad(), refGrad);

        // idempotent, and the externally owned arrays are never written back
        for (int pass = 0; pass < 3; ++pass)
        {
            correctAll();

            auto values = domainVector.boundaryData().value().copyToHost();
            auto fractions = domainVector.boundaryData().valueFraction().copyToHost();
            auto refValues = domainVector.boundaryData().refValue().copyToHost();
            auto refGrads = domainVector.boundaryData().refGrad().copyToHost();
            for (NeoN::localIdx i = 0; i < nBnd; ++i)
            {
                const auto f = fraction(i);
                const auto expected =
                    f * refValue(i) + (1 - f) * (internal + refGrad(i) / delta[i]);
                REQUIRE_THAT(values.view()[i], Catch::Matchers::WithinAbs(expected, 1e-9));
                REQUIRE(fractions.view()[i] == f);
                REQUIRE(refValues.view()[i] == refValue(i));
                REQUIRE(refGrads.view()[i] == refGrad(i));
            }
        }
    }

    SECTION("an externally switched valueFraction follows inflow and outflow" + execName)
    {
        // The inletOutlet use case: the caller flips valueFraction between 1 (inflow, fixed
        // value) and 0 (outflow, zero gradient) and the internal field moves in between.
        NeoN::fill(domainVector.boundaryData().refValue(), 300.0);

        NeoN::fill(domainVector.boundaryData().valueFraction(), 1.0);
        correctAll();
        {
            auto values = domainVector.boundaryData().value().copyToHost();
            for (NeoN::localIdx i = 0; i < nBnd; ++i)
            {
                REQUIRE_THAT(values.view()[i], Catch::Matchers::WithinAbs(300.0, 1e-12));
            }
        }

        NeoN::fill(domainVector.boundaryData().valueFraction(), 0.0);
        NeoN::fill(domainVector.internalVector(), 5.0);
        correctAll();
        {
            auto values = domainVector.boundaryData().value().copyToHost();
            for (NeoN::localIdx i = 0; i < nBnd; ++i)
            {
                REQUIRE_THAT(values.view()[i], Catch::Matchers::WithinAbs(5.0, 1e-12));
            }
        }
    }
}
