// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

TEST_CASE("fixedMean_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // 3 x 2 x 2 cells. Patch 0 is the x = 0 patch: 4 faces, each with a different adjacent cell,
    // so the adjacent cell values (set per cell below) differ from face to face.
    auto mesh = NeoN::create3DUniformMesh(exec, 3, 2, 2, 1.5, 1.0, 0.5);
    const auto offsets = mesh.boundaryMesh().offset();
    const std::pair<NeoN::localIdx, NeoN::localIdx> patch0Range {offsets[0], offsets[1]};
    const NeoN::scalar untouched = -7.0;

    auto [areasH, ownersH] =
        copyToHosts(mesh.boundaryMesh().faceAreas(), mesh.boundaryMesh().faceOwners());
    const auto areas = areasH.view();
    const auto owners = ownersH.view();

    // A field whose cell c holds cellValue(c); boundary values start as `untouched` everywhere.
    auto makeField = [&](auto cellValue)
    {
        std::vector<NeoN::scalar> cells(static_cast<std::size_t>(mesh.nCells()));
        for (std::size_t c = 0; c < cells.size(); ++c)
            cells[c] = cellValue(static_cast<NeoN::scalar>(c));
        NeoN::Field<NeoN::scalar> field(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        field.internalVector() = NeoN::Vector<NeoN::scalar>(exec, cells);
        NeoN::fill(field.boundaryData().value(), untouched);
        return field;
    };

    auto makeBoundary = [&](NeoN::scalar meanValue)
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("fixedMean"));
        dict.insert("meanValue", meanValue);
        return fvcc::VolumeBoundaryFactory<NeoN::scalar>::create("fixedMean", mesh, dict, 0);
    };

    // area weighted mean of the owner values of the patch faces, on the host
    auto ownerMean = [&](const NeoN::Field<NeoN::scalar>& field)
    {
        auto internalH = field.internalVector().copyToHost();
        NeoN::scalar weighted = 0.0, total = 0.0;
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            weighted += areas[i] * internalH.view()[owners[i]];
            total += areas[i];
        }
        return weighted / total;
    };

    auto patchMean = [&](const NeoN::Vector<NeoN::scalar>& values)
    {
        auto valuesH = values.copyToHost();
        NeoN::scalar weighted = 0.0, total = 0.0;
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            weighted += areas[i] * valuesH.view()[i];
            total += areas[i];
        }
        return weighted / total;
    };

    SECTION("a patch mean above half the target scales the owner values " + execName)
    {
        // cell values 99000 + 500 c: the patch mean is close to the target 100000, so the values
        // are scaled by meanValue / meanPsi and keep their relative differences
        const NeoN::scalar meanValue = 100000.0;
        auto field = makeField([](NeoN::scalar c) { return 99000.0 + 500.0 * c; });
        const NeoN::scalar meanPsi = ownerMean(field);
        REQUIRE(meanPsi > 0.5 * meanValue);
        REQUIRE(meanPsi != Catch::Approx(meanValue).epsilon(1e-9)); // the scaling has to do work

        auto boundary = makeBoundary(meanValue);
        REQUIRE(boundary->attributes().fixesValue == true);
        REQUIRE(boundary->attributes().assignable == false);

        boundary->correctBoundaryCondition(field);

        auto [valuesH, refValuesH, refGradH, valueFractionH] = copyToHosts(
            field.boundaryData().value(),
            field.boundaryData().refValue(),
            field.boundaryData().refGrad(),
            field.boundaryData().valueFraction()
        );
        auto internalH = field.internalVector().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            REQUIRE(
                valuesH.view()[i]
                == Catch::Approx(internalH.view()[owners[i]] * meanValue / meanPsi)
            );
            REQUIRE(refValuesH.view()[i] == valuesH.view()[i]);
            REQUIRE(valueFractionH.view()[i] == Catch::Approx(1.0));
            REQUIRE(refGradH.view()[i] == Catch::Approx(0.0));
        }
        REQUIRE(patchMean(field.boundaryData().value()) == Catch::Approx(meanValue).epsilon(1e-12));
    }

    SECTION("a patch mean at or below half the target shifts the owner values " + execName)
    {
        // cell values 10 + c: far below the target, so every value is shifted by meanValue -
        // meanPsi
        const NeoN::scalar meanValue = 100000.0;
        auto field = makeField([](NeoN::scalar c) { return 10.0 + c; });
        const NeoN::scalar meanPsi = ownerMean(field);
        REQUIRE(meanPsi <= 0.5 * meanValue);

        makeBoundary(meanValue)->correctBoundaryCondition(field);

        auto valuesH = field.boundaryData().value().copyToHost();
        auto internalH = field.internalVector().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            REQUIRE(
                valuesH.view()[i]
                == Catch::Approx(internalH.view()[owners[i]] + (meanValue - meanPsi)).epsilon(1e-12)
            );
        }
        REQUIRE(patchMean(field.boundaryData().value()) == Catch::Approx(meanValue).epsilon(1e-12));
    }

    SECTION("a zero target mean shifts the owner values " + execName)
    {
        const NeoN::scalar meanValue = 0.0;
        auto field = makeField([](NeoN::scalar c) { return 5.0 + c; });
        const NeoN::scalar meanPsi = ownerMean(field);

        makeBoundary(meanValue)->correctBoundaryCondition(field);

        auto valuesH = field.boundaryData().value().copyToHost();
        auto internalH = field.internalVector().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            REQUIRE(
                valuesH.view()[i]
                == Catch::Approx(internalH.view()[owners[i]] - meanPsi).margin(1e-12)
            );
        }
        REQUIRE(patchMean(field.boundaryData().value()) == Catch::Approx(0.0).margin(1e-12));
    }

    SECTION(
        "the scaling uses magnitudes, like OpenFOAM, so a negative mean keeps its sign " + execName
    )
    {
        // meanPsi is about -80, |meanPsi| = 80 > 0.5 * 100: values are multiplied by 100 / 80, a
        // positive factor, so the patch mean is about -100 and not +100
        const NeoN::scalar meanValue = 100.0;
        auto field = makeField([](NeoN::scalar c) { return -80.0 - 0.5 * c; });
        const NeoN::scalar meanPsi = ownerMean(field);
        REQUIRE(meanPsi < 0.0);
        REQUIRE(Kokkos::abs(meanPsi) > 0.5 * meanValue);

        makeBoundary(meanValue)->correctBoundaryCondition(field);

        auto valuesH = field.boundaryData().value().copyToHost();
        auto internalH = field.internalVector().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            REQUIRE(
                valuesH.view()[i]
                == Catch::Approx(internalH.view()[owners[i]] * meanValue / Kokkos::abs(meanPsi))
            );
        }
        REQUIRE(
            patchMean(field.boundaryData().value()) == Catch::Approx(-meanValue).epsilon(1e-12)
        );
    }

    SECTION("only the faces of the patch are written, and the values follow the cells " + execName)
    {
        const NeoN::scalar meanValue = 100000.0;
        auto field = makeField([](NeoN::scalar c) { return 99000.0 + 500.0 * c; });
        auto boundary = makeBoundary(meanValue);
        boundary->correctBoundaryCondition(field);

        auto valuesH = field.boundaryData().value().copyToHost();
        for (auto i = offsets[1]; i < offsets.back(); ++i)
            REQUIRE(valuesH.view()[i] == untouched);

        // recomputed from the current cell values on every call
        NeoN::fill(field.internalVector(), 101000.0);
        boundary->correctBoundaryCondition(field);
        auto valuesAfterH = field.boundaryData().value().copyToHost();
        bool changed = false;
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            changed = changed || valuesAfterH.view()[i] != Catch::Approx(valuesH.view()[i]);
            // uniform cells: every face is scaled to the target
            REQUIRE(valuesAfterH.view()[i] == Catch::Approx(meanValue));
        }
        REQUIRE(changed);
    }

    SECTION("evaluateOnly keeps the values of the last update " + execName)
    {
        // OpenFOAM updates fixedMean when the matrix is built; the evaluation after the solve
        // keeps those values although the cells have changed in between
        const NeoN::scalar meanValue = 100000.0;
        auto field = makeField([](NeoN::scalar c) { return 99000.0 + 500.0 * c; });
        auto boundary = makeBoundary(meanValue);
        boundary->correctBoundaryCondition(field);
        auto before = field.boundaryData().value().copyToHost();

        NeoN::fill(field.internalVector(), 101000.0);
        fvcc::BoundaryContext frozen;
        frozen.insertValue("evaluateOnly", 1.0);
        boundary->correctBoundaryCondition(field, frozen);
        auto kept = field.boundaryData().value().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
            REQUIRE(kept.view()[i] == before.view()[i]);

        // a context without the key, or with it at zero, recomputes from the new cells
        fvcc::BoundaryContext open;
        open.insertValue("evaluateOnly", 0.0);
        boundary->correctBoundaryCondition(field, open);
        auto after = field.boundaryData().value().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
            REQUIRE(after.view()[i] == Catch::Approx(meanValue));
        REQUIRE(after.view()[patch0Range.first] != Catch::Approx(before.view()[patch0Range.first]));
    }

    SECTION("a missing meanValue is rejected " + execName)
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("fixedMean"));
        REQUIRE_THROWS_AS(
            fvcc::VolumeBoundaryFactory<NeoN::scalar>::create("fixedMean", mesh, dict, 0),
            NeoN::NeoNException
        );
    }
}
