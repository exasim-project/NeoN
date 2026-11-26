// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

// mapEnergyBoundaryFromTemperature is the C++-side replacement for the Python-side
// he = Cv*(T - Tref) boundary hack validated earlier (neorhoPimpleFoam_investigation_summary.md
// §4.40): a 28.6x T-spread reduction on the real duct case once the energy equation is solved
// for he instead of raw T. These tests pin the mapping itself -- the affine relation applied to
// both halves of the mixed-BC representation (refValue/refGrad/valueFraction) -- independent of
// which concrete BC type (FixedEnergy, GradientEnergy) ends up consuming it.
TEST_CASE("mapEnergyBoundaryFromTemperature")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = NeoN::createSingleCellMesh(exec);
    const auto nBnd = mesh.nBoundaryFaces();
    const NeoN::scalar Cv = 712.0;
    const NeoN::scalar Tref = 298.15;

    NeoN::Field<NeoN::scalar> T(exec, mesh.nCells(), mesh.boundaryMesh().offset());
    NeoN::Field<NeoN::scalar> he(exec, mesh.nCells(), mesh.boundaryMesh().offset());

    SECTION("fixedValue-type T boundary (refValue) maps to the real literal he value" + execName)
    {
        // The duct case's actual inlet condition: T fixedValue(300). Literal OpenFOAM's
        // fixedEnergyFvPatchScalarField computes he = Cv*(T-Tstd) = 712*(300-298.15) =
        // 1317.2 exactly (already confirmed against the live solver this investigation,
        // §4.40) -- this test pins that same number from the mapping function alone.
        NeoN::fill(T.boundaryData().refValue(), 300.0);
        NeoN::fill(T.boundaryData().refGrad(), 0.0);
        NeoN::fill(T.boundaryData().valueFraction(), 1.0); // Dirichlet-type

        NeoN::finiteVolume::cellCentred::mapEnergyBoundaryFromTemperature(
            T.boundaryData(), he.boundaryData(), Cv, Tref
        );

        auto heRefValue = he.boundaryData().refValue().copyToHost();
        auto heValueFraction = he.boundaryData().valueFraction().copyToHost();
        auto heRefValueView = heRefValue.view();
        auto heValueFractionView = heValueFraction.view();
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            REQUIRE_THAT(heRefValueView[i], Catch::Matchers::WithinAbs(1317.2, 1e-9));
            REQUIRE(heValueFractionView[i] == 1.0);
        }
    }

    SECTION("zeroGradient-type T boundary (refGrad=0) maps to zeroGradient on he" + execName)
    {
        // The duct case's walls/outlet condition: T zeroGradient. he.refGrad = Cv*T.refGrad
        // = Cv*0 = 0 exactly -- gradientEnergy collapses to true zero-gradient for this
        // constant, spatially-uniform Cv case (§4.40's stated simplification over OpenFOAM's
        // general temperature-dependent-Cv chain rule).
        NeoN::fill(T.boundaryData().refValue(), 0.0);
        NeoN::fill(T.boundaryData().refGrad(), 0.0);
        NeoN::fill(T.boundaryData().valueFraction(), 0.0); // Neumann-type

        NeoN::finiteVolume::cellCentred::mapEnergyBoundaryFromTemperature(
            T.boundaryData(), he.boundaryData(), Cv, Tref
        );

        auto heRefGrad = he.boundaryData().refGrad().copyToHost();
        auto heValueFraction = he.boundaryData().valueFraction().copyToHost();
        auto heRefGradView = heRefGrad.view();
        auto heValueFractionView = heValueFraction.view();
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            REQUIRE(heRefGradView[i] == 0.0);
            REQUIRE(heValueFractionView[i] == 0.0);
        }
    }

    SECTION("a genuinely nonzero T gradient maps through the Cv chain rule" + execName)
    {
        // Not this investigation's actual case, but pins the general (non-collapsed) formula:
        // he.refGrad = Cv * T.refGrad, not just the degenerate zero case above.
        NeoN::fill(T.boundaryData().refGrad(), 2.0);
        NeoN::fill(T.boundaryData().valueFraction(), 0.0);

        NeoN::finiteVolume::cellCentred::mapEnergyBoundaryFromTemperature(
            T.boundaryData(), he.boundaryData(), Cv, Tref
        );

        auto heRefGrad = he.boundaryData().refGrad().copyToHost();
        auto heRefGradView = heRefGrad.view();
        for (NeoN::localIdx i = 0; i < nBnd; ++i)
        {
            REQUIRE_THAT(heRefGradView[i], Catch::Matchers::WithinAbs(Cv * 2.0, 1e-9));
        }
    }

    SECTION(
        "end-to-end: FixedEnergy + GradientEnergy reproduce literal's real duct-case "
        "boundary he values after the mapping"
        + execName
    )
    {
        // Patch 0 behaves like the inlet (T fixedValue 300), the rest like walls/outlet
        // (T zeroGradient) -- exercises the full real pipeline: map onto he's boundary,
        // then let he's own BC objects (FixedEnergy/GradientEnergy) materialise value()
        // from what was just written, exactly as a solver would call it every outer
        // iteration.
        auto TRefValue = T.boundaryData().refValue().copyToHost();
        auto TRefGrad = T.boundaryData().refGrad().copyToHost();
        auto TValueFraction = T.boundaryData().valueFraction().copyToHost();
        std::vector<NeoN::scalar> refValueSeed(static_cast<sizeType>(nBnd), 0.0);
        std::vector<NeoN::scalar> refGradSeed(static_cast<sizeType>(nBnd), 0.0);
        std::vector<NeoN::scalar> fractionSeed(static_cast<sizeType>(nBnd), 0.0);
        refValueSeed[0] = 300.0;
        fractionSeed[0] = 1.0;
        T.boundaryData().refValue() = NeoN::Vector<NeoN::scalar>(exec, refValueSeed);
        T.boundaryData().refGrad() = NeoN::Vector<NeoN::scalar>(exec, refGradSeed);
        T.boundaryData().valueFraction() = NeoN::Vector<NeoN::scalar>(exec, fractionSeed);

        NeoN::finiteVolume::cellCentred::mapEnergyBoundaryFromTemperature(
            T.boundaryData(), he.boundaryData(), Cv, Tref
        );

        NeoN::Dictionary dict;
        std::vector<
            std::unique_ptr<NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>>>
            boundaries;
        for (NeoN::localIdx patchID = 0; patchID < mesh.nBoundaries(); ++patchID)
        {
            const std::string type = (patchID == 0) ? "fixedEnergy" : "gradientEnergy";
            boundaries.push_back(
                NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                    type, mesh, dict, patchID
                )
            );
        }
        NeoN::fill(he.internalVector(), 1317.2); // he's internal value already at the inlet value
        for (auto& bc : boundaries)
        {
            bc->correctBoundaryCondition(he);
        }

        auto heValue = he.boundaryData().value().copyToHost();
        auto heValueView = heValue.view();
        REQUIRE_THAT(heValueView[0], Catch::Matchers::WithinAbs(1317.2, 1e-9));
        for (NeoN::localIdx i = 1; i < nBnd; ++i)
        {
            // gradientEnergy with refGrad=0: value == owner internal value (true zeroGradient)
            REQUIRE_THAT(heValueView[i], Catch::Matchers::WithinAbs(1317.2, 1e-9));
        }
    }
}
