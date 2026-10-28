// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

TEST_CASE("flowRateInletVelocity_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // 3 x 2 x 2 cells. Patch 0 is the x = 0 patch: 4 faces of area 0.125, outward normal (-1 0 0),
    // each with a different adjacent cell.
    auto mesh = NeoN::create3DUniformMesh(exec, 3, 2, 2, 1.5, 1.0, 0.5);
    const auto offsets = mesh.boundaryMesh().offset();
    const std::pair<NeoN::localIdx, NeoN::localIdx> patch0Range {offsets[0], offsets[1]};
    const NeoN::scalar untouched = -7.0;

    auto [areasH, normalsH, ownersH] = copyToHosts(
        mesh.boundaryMesh().faceAreas(),
        mesh.boundaryMesh().faceNormals(),
        mesh.boundaryMesh().faceOwners()
    );
    const auto areas = areasH.view();
    const auto normals = normalsH.view();
    const auto owners = ownersH.view();

    auto unitNormal = [&](NeoN::localIdx i) { return normals[i] / areas[i]; };

    NeoN::scalar patchArea = 0.0;
    for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        patchArea += areas[i];

    // The patch is the inlet at x = 0: the outward normal points to -x
    for (auto i = patch0Range.first; i < patch0Range.second; ++i)
    {
        REQUIRE(unitNormal(i)[0] == Catch::Approx(-1.0));
        REQUIRE(unitNormal(i)[1] == Catch::Approx(0.0).margin(1e-14));
        REQUIRE(unitNormal(i)[2] == Catch::Approx(0.0).margin(1e-14));
    }

    // A velocity field whose cell c holds cellVelocity(c); boundary values start as `untouched`.
    auto makeField = [&](auto cellVelocity)
    {
        std::vector<NeoN::Vec3> cells(static_cast<std::size_t>(mesh.nCells()));
        for (std::size_t c = 0; c < cells.size(); ++c)
            cells[c] = cellVelocity(static_cast<NeoN::scalar>(c));
        NeoN::Field<NeoN::Vec3> field(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        field.internalVector() = NeoN::Vector<NeoN::Vec3>(exec, cells);
        NeoN::fill(field.boundaryData().value(), NeoN::Vec3 {untouched, untouched, untouched});
        return field;
    };
    auto zeroVelocity = [](NeoN::scalar) { return NeoN::Vec3 {0.0, 0.0, 0.0}; };

    // x component of the boundary value of the first face of patch 0 (the host copy has to outlive
    // the view)
    auto firstValueX = [&](const NeoN::Field<NeoN::Vec3>& field)
    {
        auto valuesH = field.boundaryData().value().copyToHost();
        return valuesH.view()[patch0Range.first][0];
    };

    // A density field with a given boundary value per boundary face (nBoundaryFaces entries)
    auto makeDensity = [&](const std::vector<NeoN::scalar>& boundaryRho)
    {
        auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
        fvcc::VolumeField<NeoN::scalar> rho(exec, "rho", mesh, bcs);
        NeoN::fill(rho.internalVector(), 1.0);
        rho.boundaryData().value() = NeoN::Vector<NeoN::scalar>(exec, boundaryRho);
        return rho;
    };
    std::vector<NeoN::scalar> rhoPerFace(static_cast<std::size_t>(mesh.nBoundaryFaces()), 1.0);
    for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        rhoPerFace[static_cast<std::size_t>(i)] = 1.0 + 0.25 * static_cast<NeoN::scalar>(i);

    auto makeBoundary = [&](NeoN::Dictionary dict)
    {
        dict.insert("type", std::string("flowRateInletVelocity"));
        return fvcc::VolumeBoundaryFactory<NeoN::Vec3>::create(
            "flowRateInletVelocity", mesh, dict, 0
        );
    };

    // flow rate carried through the patch by the boundary values: -sum(rho magSf n & U)
    auto flowRateOf = [&](const NeoN::Field<NeoN::Vec3>& field, auto rhoOf)
    {
        auto valuesH = field.boundaryData().value().copyToHost();
        NeoN::scalar flow = 0.0;
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
            flow -= rhoOf(i) * areas[i] * (unitNormal(i) & valuesH.view()[i]);
        return flow;
    };

    SECTION("a volumetric flow rate gives a uniform velocity along the inward normal " + execName)
    {
        const NeoN::scalar flowRate = 0.2;
        NeoN::Dictionary dict;
        dict.insert("volumetricFlowRate", flowRate);
        auto boundary = makeBoundary(dict);
        REQUIRE(boundary->attributes().fixesValue == true);
        // not assignable, so constrainHbyA copies the prescribed velocity into HbyA and the inlet
        // flux reaches the pressure equation
        REQUIRE(boundary->attributes().assignable == false);

        auto field = makeField(zeroVelocity);
        boundary->correctBoundaryCondition(field); // no context needed

        auto [valuesH, refValuesH, valueFractionH, refGradH] = copyToHosts(
            field.boundaryData().value(),
            field.boundaryData().refValue(),
            field.boundaryData().valueFraction(),
            field.boundaryData().refGrad()
        );
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            REQUIRE(valuesH.view()[i][0] == Catch::Approx(flowRate / patchArea));
            REQUIRE(valuesH.view()[i][1] == Catch::Approx(0.0).margin(1e-14));
            REQUIRE(valuesH.view()[i][2] == Catch::Approx(0.0).margin(1e-14));
            REQUIRE(refValuesH.view()[i][0] == valuesH.view()[i][0]);
            REQUIRE(valueFractionH.view()[i] == Catch::Approx(1.0));
            REQUIRE(refGradH.view()[i][0] == Catch::Approx(0.0));
        }
        REQUIRE(flowRateOf(field, [](NeoN::localIdx) { return 1.0; }) == Catch::Approx(flowRate));

        // the context makes no difference for a volumetric flow rate
        auto fieldCtx = makeField(zeroVelocity);
        fvcc::BoundaryContext ctx;
        boundary->correctBoundaryCondition(fieldCtx, ctx);
        auto valuesCtxH = fieldCtx.boundaryData().value().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
            REQUIRE(valuesCtxH.view()[i][0] == valuesH.view()[i][0]);

        // other patches are not written
        for (auto i = offsets[1]; i < offsets.back(); ++i)
            REQUIRE(valuesH.view()[i][0] == untouched);
    }

    SECTION("a mass flow rate uses the density of the boundary faces from the context " + execName)
    {
        const NeoN::scalar mdot = 1.0e-4;
        NeoN::Dictionary dict;
        dict.insert("massFlowRate", mdot);
        auto boundary = makeBoundary(dict);

        auto rho = makeDensity(rhoPerFace);
        fvcc::BoundaryContext ctx;
        ctx.insert("rho", rho);

        auto field = makeField(zeroVelocity);
        boundary->correctBoundaryCondition(field, ctx);

        NeoN::scalar rhoArea = 0.0;
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
            rhoArea += rhoPerFace[static_cast<std::size_t>(i)] * areas[i];

        auto valuesH = field.boundaryData().value().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
            REQUIRE(
                valuesH.view()[i][0] == Catch::Approx(mdot / rhoArea)
            ); // the same velocity on every face
        REQUIRE(
            flowRateOf(
                field, [&](NeoN::localIdx i) { return rhoPerFace[static_cast<std::size_t>(i)]; }
            )
            == Catch::Approx(mdot)
        );
    }

    SECTION("the density field wins over rhoInlet, rhoInlet is used without a field " + execName)
    {
        const NeoN::scalar mdot = 1.0e-4;
        NeoN::Dictionary dict;
        dict.insert("massFlowRate", mdot);
        dict.insert("rhoInlet", NeoN::scalar(5.0));
        auto boundary = makeBoundary(dict);

        auto rho = makeDensity(rhoPerFace);
        fvcc::BoundaryContext ctxWithRho;
        ctxWithRho.insert("rho", rho);
        fvcc::BoundaryContext ctxWithoutRho;

        NeoN::scalar rhoArea = 0.0;
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
            rhoArea += rhoPerFace[static_cast<std::size_t>(i)] * areas[i];

        auto fieldA = makeField(zeroVelocity);
        boundary->correctBoundaryCondition(fieldA, ctxWithRho);
        REQUIRE(firstValueX(fieldA) == Catch::Approx(mdot / rhoArea));

        auto fieldB = makeField(zeroVelocity);
        boundary->correctBoundaryCondition(fieldB, ctxWithoutRho);
        REQUIRE(firstValueX(fieldB) == Catch::Approx(mdot / (5.0 * patchArea)));

        // and also without any context
        auto fieldC = makeField(zeroVelocity);
        boundary->correctBoundaryCondition(fieldC);
        REQUIRE(firstValueX(fieldC) == Catch::Approx(mdot / (5.0 * patchArea)));
    }

    SECTION("rho none means density 1 " + execName)
    {
        const NeoN::scalar mdot = 0.3;
        NeoN::Dictionary dict;
        dict.insert("massFlowRate", mdot);
        dict.insert("rho", std::string("none"));
        auto boundary = makeBoundary(dict);

        auto rho = makeDensity(rhoPerFace); // present, but ignored
        fvcc::BoundaryContext ctx;
        ctx.insert("rho", rho);

        auto field = makeField(zeroVelocity);
        boundary->correctBoundaryCondition(field, ctx);
        REQUIRE(firstValueX(field) == Catch::Approx(mdot / patchArea));
    }

    SECTION(
        "a mass flow rate without any density is an error with a context, zero-gradient without "
        + execName
    )
    {
        NeoN::Dictionary dict;
        dict.insert("massFlowRate", NeoN::scalar(1.0e-4));
        auto boundary = makeBoundary(dict);

        auto field = makeField([](NeoN::scalar c) { return NeoN::Vec3 {0.1 * c, 0.0, 0.0}; });

        fvcc::BoundaryContext emptyCtx;
        REQUIRE_THROWS_AS(boundary->correctBoundaryCondition(field, emptyCtx), NeoN::NeoNException);

        boundary->correctBoundaryCondition(field); // plain overload: zero-gradient
        auto [valuesH, valueFractionH, internalH] = copyToHosts(
            field.boundaryData().value(),
            field.boundaryData().valueFraction(),
            field.internalVector()
        );
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            REQUIRE(valuesH.view()[i][0] == Catch::Approx(internalH.view()[owners[i]][0]));
            REQUIRE(valueFractionH.view()[i] == Catch::Approx(0.0));
        }
    }

    SECTION(
        "extrapolateProfile scales the inflow of the adjacent cells to the flow rate " + execName
    )
    {
        // Inflow Ux = 0.5 + 0.1 c except for the cells with c > 7, which have reverse flow; Uy, Uz
        // are tangential. The estimated flow rate is above 0.5 flowRate, so the inflow is scaled.
        const NeoN::scalar flowRate = 0.2;
        NeoN::Dictionary dict;
        dict.insert("volumetricFlowRate", flowRate);
        dict.insert("extrapolateProfile", true);
        auto boundary = makeBoundary(dict);

        auto cellVelocity = [](NeoN::scalar c) {
            return NeoN::Vec3 {c > 7.0 ? -0.2 : 0.5 + 0.1 * c, 0.05 * c, -0.03 * c};
        };
        auto field = makeField(cellVelocity);
        boundary->correctBoundaryCondition(field);

        // reference computed on the host from the formulas
        std::vector<NeoN::scalar> nUp;
        NeoN::scalar estimated = 0.0;
        bool anyReverseFlow = false;
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            const auto c = static_cast<NeoN::scalar>(owners[i]);
            const NeoN::scalar normalVelocity = unitNormal(i) & cellVelocity(c);
            anyReverseFlow = anyReverseFlow || normalVelocity > 0.0;
            nUp.push_back(Kokkos::min(normalVelocity, 0.0));
            estimated -= areas[i] * nUp.back();
        }
        REQUIRE(anyReverseFlow); // the reverse flow removal has to be exercised
        REQUIRE(estimated > 0.5 * flowRate);
        REQUIRE(estimated != Catch::Approx(flowRate).epsilon(1e-6)); // the scaling has to do work

        auto valuesH = field.boundaryData().value().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            const auto c = static_cast<NeoN::scalar>(owners[i]);
            const auto cellVelocityAtFace = cellVelocity(c);
            const auto n = unitNormal(i);
            const auto tangential = cellVelocityAtFace - n * (n & cellVelocityAtFace);
            const NeoN::scalar nUpNew =
                nUp[static_cast<std::size_t>(i - patch0Range.first)] * flowRate / estimated;
            const auto expected = tangential + nUpNew * n;
            for (int k = 0; k < 3; ++k)
                REQUIRE(valuesH.view()[i][k] == Catch::Approx(expected[k]).margin(1e-14));
            // the tangential velocity of the adjacent cell is kept
            const auto got = valuesH.view()[i];
            REQUIRE((got - n * (n & got))[1] == Catch::Approx(tangential[1]).margin(1e-14));
        }
        REQUIRE(flowRateOf(field, [](NeoN::localIdx) { return 1.0; }) == Catch::Approx(flowRate));
    }

    SECTION("extrapolateProfile shifts a small inflow up to the flow rate " + execName)
    {
        // Inflow Ux = 0.01 only: the estimated flow rate (0.005) is below 0.5 flowRate, so every
        // face is shifted by (flowRate - estimated) / sum(magSf)
        const NeoN::scalar flowRate = 0.2;
        NeoN::Dictionary dict;
        dict.insert("volumetricFlowRate", flowRate);
        dict.insert("extrapolateProfile", true);
        auto boundary = makeBoundary(dict);

        auto cellVelocity = [](NeoN::scalar c) { return NeoN::Vec3 {0.01, 0.05 * c, 0.0}; };
        auto field = makeField(cellVelocity);
        boundary->correctBoundaryCondition(field);

        const NeoN::scalar estimated = 0.01 * patchArea;
        REQUIRE(estimated < 0.5 * flowRate);
        const NeoN::scalar shift = (flowRate - estimated) / patchArea;

        auto valuesH = field.boundaryData().value().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
        {
            const auto c = static_cast<NeoN::scalar>(owners[i]);
            const auto cellVelocityAtFace = cellVelocity(c);
            const auto n = unitNormal(i);
            const auto expected = (cellVelocityAtFace - n * (n & cellVelocityAtFace))
                                + (n & cellVelocityAtFace) * n - shift * n; // nUp - shift
            for (int k = 0; k < 3; ++k)
                REQUIRE(valuesH.view()[i][k] == Catch::Approx(expected[k]).margin(1e-14));
        }
        REQUIRE(flowRateOf(field, [](NeoN::localIdx) { return 1.0; }) == Catch::Approx(flowRate));
    }

    SECTION("a flow rate is required " + execName)
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("flowRateInletVelocity"));
        REQUIRE_THROWS_AS(
            fvcc::VolumeBoundaryFactory<NeoN::Vec3>::create("flowRateInletVelocity", mesh, dict, 0),
            NeoN::NeoNException
        );
    }
}
