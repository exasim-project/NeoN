// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include <cmath>

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace
{

// plenumPressure of the helmholtzResonance tutorial, with a timeScale that can be set
struct PlenumParameters
{
    NeoN::scalar gamma = 1.4;
    NeoN::scalar R = 287.04;
    NeoN::scalar supplyMassFlowRate = 1.0e-4;
    NeoN::scalar supplyTotalTemperature = 300.0;
    NeoN::scalar plenumVolume = 1.25e-4;
    NeoN::scalar plenumDensity = 1.1613;
    NeoN::scalar plenumTemperature = 300.0;
    NeoN::scalar inletAreaRatio = 1.0;
    NeoN::scalar dischargeCoefficient = 0.8;
    NeoN::scalar timeScale = 1.0e-4;

    NeoN::Dictionary dict() const
    {
        NeoN::Dictionary d;
        d.insert("type", std::string("plenumPressure"));
        d.insert("gamma", gamma);
        d.insert("R", R);
        d.insert("supplyMassFlowRate", supplyMassFlowRate);
        d.insert("supplyTotalTemperature", supplyTotalTemperature);
        d.insert("plenumVolume", plenumVolume);
        d.insert("plenumDensity", plenumDensity);
        d.insert("plenumTemperature", plenumTemperature);
        d.insert("inletAreaRatio", inletAreaRatio);
        d.insert("inletDischargeCoefficient", dischargeCoefficient);
        d.insert("timeScale", timeScale);
        d.insert("value", NeoN::scalar(100000.0));
        return d;
    }
};

// A hand-written copy of OpenFOAM's plenumPressureFvPatchScalarField::updateCoeffs() on plain
// numbers: the plenum state that persists between updates, and the patch values of one update.
struct Reference
{
    PlenumParameters par;
    NeoN::scalar rho, rhoOld, T, TOld;
    int timeIndex = -1;

    explicit Reference(const PlenumParameters& p)
        : par(p), rho(p.plenumDensity), rhoOld(p.plenumDensity), T(p.plenumTemperature),
          TOld(p.plenumTemperature)
    {}

    std::vector<NeoN::scalar> update(
        int index,
        NeoN::scalar dt,
        const std::vector<NeoN::scalar>& phi,
        const std::vector<NeoN::Vec3>& U,
        const std::vector<NeoN::scalar>& pOld,
        const std::vector<NeoN::scalar>& p
    )
    {
        if (index != timeIndex)
        {
            timeIndex = index;
            rhoOld = rho;
            TOld = T;
        }
        NeoN::scalar sum = 0;
        for (auto v : phi)
            sum += v;
        const NeoN::scalar m = -sum;
        const NeoN::scalar cv = par.R / (par.gamma - 1), cp = par.R * par.gamma / (par.gamma - 1);
        rho = rhoOld + dt / par.plenumVolume * (par.supplyMassFlowRate - m);
        T = TOld
          + dt / (rho * cv * par.plenumVolume)
                * (par.supplyMassFlowRate * (cp * par.supplyTotalTemperature - cv * T)
                   - m * par.R * T);
        const NeoN::scalar pPlenum = rho * par.R * T;

        const NeoN::scalar oneByFraction = par.timeScale / dt;
        const NeoN::scalar fraction = oneByFraction < 1.0 ? 1.0 : 1.0 / oneByFraction;
        std::vector<NeoN::scalar> out(phi.size());
        for (std::size_t i = 0; i < phi.size(); ++i)
        {
            const NeoN::Vec3 exitVelocity = U[i] / par.inletAreaRatio;
            const NeoN::scalar uE = exitVelocity[0] * exitVelocity[0]
                                  + exitVelocity[1] * exitVelocity[1]
                                  + exitVelocity[2] * exitVelocity[2];
            const NeoN::scalar r = 1.0 - (par.gamma - 1.0) * uE / (2.0 * par.gamma * par.R * T);
            const NeoN::scalar a =
                (1.0 - r) / (r * r * par.dischargeCoefficient * par.dischargeCoefficient);
            const NeoN::scalar s = 2.0 / (1.0 + std::sqrt(1.0 + 4.0 * a));
            const NeoN::scalar t = std::pow(s, par.gamma / (par.gamma - 1.0));
            const NeoN::scalar pNew = phi[i] >= 0 ? std::max(p[i], pPlenum) : t * pPlenum;
            out[i] = pOld[i] + fraction * (pNew - pOld[i]);
        }
        return out;
    }
};

} // namespace

TEST_CASE("plenumPressure_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // 3 x 2 x 2 cells; patch 0 is x = 0 with 4 faces, the plenum patch
    auto mesh = NeoN::create3DUniformMesh(exec, 3, 2, 2, 1.5, 1.0, 0.5);
    const auto offsets = mesh.boundaryMesh().offset();
    const auto first = offsets[0];
    const NeoN::scalar untouched = 5.0;

    // per-face data of patch 0: both flow directions, different exit velocities and histories
    const std::vector<NeoN::scalar> phiA {-2.0e-5, -3.0e-5, 1.0e-5, -4.0e-5};
    const std::vector<NeoN::scalar> phiB {-1.0e-5, -5.0e-5, 2.0e-5, -3.0e-5};
    const std::vector<NeoN::Vec3> uPatch {
        {0.2, 0.0, 0.0}, {30.0, 5.0, 0.0}, {0.0, 0.0, 0.0}, {80.0, 0.0, 10.0}
    };
    const std::vector<NeoN::scalar> pOldPatch {100000.0, 100003.0, 99990.0, 100010.0};
    // current boundary values: the third face is above the plenum pressure, the others below
    const std::vector<NeoN::scalar> pNowPatch {100000.0, 100001.0, 100500.0, 99999.0};
    const NeoN::scalar dt = 1.0e-4;

    NeoN::Database db;
    PlenumParameters par;
    auto buildP = [&](const PlenumParameters& parameters) -> fvcc::VolumeField<NeoN::scalar>&
    {
        fvcc::VectorCollection& collection = fvcc::VectorCollection::instance(db, "collection");
        auto create = [&](NeoN::Database& d)
        {
            std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs {};
            for (NeoN::localIdx patchi = 0; patchi < mesh.boundaryMesh().nBoundaries(); ++patchi)
            {
                if (patchi == 0)
                {
                    bcs.push_back(
                        fvcc::VolumeBoundary<NeoN::scalar>(mesh, parameters.dict(), patchi)
                    );
                    continue;
                }
                NeoN::Dictionary dict;
                dict.insert("type", std::string("fixedValue"));
                dict.insert("fixedValue", untouched);
                bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, dict, patchi));
            }
            NeoN::Field<NeoN::scalar> domainVector(
                mesh.exec(),
                NeoN::Vector<NeoN::scalar>(mesh.exec(), mesh.nCells(), 100000.0),
                mesh.boundaryMesh().offset()
            );
            fvcc::VolumeField<NeoN::scalar> vf(
                mesh.exec(), "p", mesh, domainVector, bcs, d, "", ""
            );
            return NeoN::Document(
                {{"name", vf.name},
                 {"timeIndex", std::int64_t(0)},
                 {"iterationIndex", std::int64_t(0)},
                 {"subCycleIndex", std::int64_t(0)},
                 {"field", vf}},
                fvcc::validateVectorDoc
            );
        };
        fvcc::VolumeField<NeoN::scalar>& p =
            collection.registerVector<fvcc::VolumeField<NeoN::scalar>>(create);
        p.correctBoundaryConditions();
        return p;
    };

    auto fillPatch0 = [&](auto& field, const auto& values)
    {
        using T = std::decay_t<decltype(values[0])>;
        std::vector<T> all(static_cast<std::size_t>(mesh.nBoundaryFaces()), T {});
        for (std::size_t k = 0; k < values.size(); ++k)
            all[static_cast<std::size_t>(first) + k] = values[k];
        field.boundaryData().value() = NeoN::Vector<T>(exec, all);
    };

    auto makeFlux = [&](const std::vector<NeoN::scalar>& values)
    {
        fvcc::SurfaceField<NeoN::scalar> phi(
            exec, "phi", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh)
        );
        NeoN::fill(phi.internalVector(), NeoN::scalar(0));
        fillPatch0(phi, values);
        return phi;
    };

    auto U = fvcc::VolumeField<NeoN::Vec3>(
        exec, "U", mesh, fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh)
    );
    NeoN::fill(U.internalVector(), NeoN::Vec3 {0.0, 0.0, 0.0});
    fillPatch0(U, uPatch);

    auto patchValues = [&](const fvcc::VolumeField<NeoN::scalar>& p)
    {
        auto h = p.boundaryData().value().copyToHost();
        return std::vector<NeoN::scalar>(h.view().data() + first, h.view().data() + first + 4);
    };

    auto prepare = [&](fvcc::VolumeField<NeoN::scalar>& p)
    {
        NeoN::fill(oldTime(p).boundaryData().value(), NeoN::scalar(0));
        fillPatch0(oldTime(p), pOldPatch);
        fillPatch0(p, pNowPatch);
    };

    auto context = [&](fvcc::BoundaryContext& ctx,
                       const fvcc::SurfaceField<NeoN::scalar>& phi,
                       const fvcc::VolumeField<NeoN::scalar>& p,
                       int timeIndex)
    {
        ctx.insert("phi", phi);
        ctx.insert("U", U);
        ctx.insert("p", p);
        ctx.insertValue("deltaT", dt);
        ctx.insertValue("timeIndex", NeoN::scalar(timeIndex));
    };

    SECTION("one update matches OpenFOAM's formulas " + execName)
    {
        auto& p = buildP(par);
        prepare(p);
        auto phi = makeFlux(phiA);
        fvcc::BoundaryContext ctx;
        context(ctx, phi, p, 1);

        Reference reference(par);
        const auto expected = reference.update(1, dt, phiA, uPatch, pOldPatch, pNowPatch);
        p.correctBoundaryConditions(ctx);

        const auto values = patchValues(p);
        for (std::size_t k = 0; k < 4; ++k)
            REQUIRE(values[k] == Catch::Approx(expected[k]).epsilon(1e-12));
        // the faces with phi < 0 are strictly below the plenum pressure, the face with phi >= 0
        // keeps the larger of its own value and the plenum pressure
        const NeoN::scalar pPlenum = reference.rho * par.R * reference.T;
        REQUIRE(values[2] == Catch::Approx(std::max(pNowPatch[2], pPlenum)).epsilon(1e-12));
        REQUIRE(values[1] < pPlenum);
        // the other patches are not touched
        auto all = p.boundaryData().value().copyToHost();
        for (auto i = offsets[1]; i < offsets.back(); ++i)
            REQUIRE(all.view()[i] == untouched);

        auto [refValueH, valueFractionH] =
            copyToHosts(p.boundaryData().refValue(), p.boundaryData().valueFraction());
        for (auto i = first; i < offsets[1]; ++i)
        {
            REQUIRE(refValueH.view()[i] == all.view()[i]);
            REQUIRE(valueFractionH.view()[i] == Catch::Approx(1.0));
        }
    }

    SECTION("a repeated update in a time step integrates from the old state " + execName)
    {
        auto& p = buildP(par);
        prepare(p);
        Reference reference(par);

        // two outer correctors of time index 1, then the first of time index 2
        auto phi1 = makeFlux(phiA);
        auto phi2 = makeFlux(phiB);
        fvcc::BoundaryContext c1, c2, c3;
        context(c1, phi1, p, 1);
        context(c2, phi2, p, 1);
        context(c3, phi1, p, 2);

        for (auto* step : {&c1, &c2, &c3})
        {
            const bool second = step == &c2;
            const int index = step == &c3 ? 2 : 1;
            const auto& phiNow = second ? phiB : phiA;
            // the pressure the patch carries before the update is the previous update's result
            const auto before = patchValues(p);
            const auto expected = reference.update(index, dt, phiNow, uPatch, pOldPatch, before);
            p.correctBoundaryConditions(*step);
            const auto values = patchValues(p);
            for (std::size_t k = 0; k < 4; ++k)
                REQUIRE(values[k] == Catch::Approx(expected[k]).epsilon(1e-12));
        }
        // the steps differ, or the check would also pass for a state that never advances
        REQUIRE(reference.rhoOld != par.plenumDensity);
    }

    SECTION("a timeScale above the time step relaxes towards the new value " + execName)
    {
        PlenumParameters slow = par;
        slow.timeScale = 4.0 * dt;
        auto& p = buildP(slow);
        prepare(p);
        auto phi = makeFlux(phiA);
        fvcc::BoundaryContext ctx;
        context(ctx, phi, p, 1);

        Reference reference(slow);
        const auto expected = reference.update(1, dt, phiA, uPatch, pOldPatch, pNowPatch);
        p.correctBoundaryConditions(ctx);
        const auto values = patchValues(p);
        for (std::size_t k = 0; k < 4; ++k)
            REQUIRE(values[k] == Catch::Approx(expected[k]).epsilon(1e-12));

        // the relaxation has to matter: the unrelaxed update is a different pressure
        Reference unrelaxed(par);
        const auto full = unrelaxed.update(1, dt, phiA, uPatch, pOldPatch, pNowPatch);
        REQUIRE(std::abs(full[1] - expected[1]) > 1e-3);
    }

    SECTION("evaluateOnly keeps the values and the state " + execName)
    {
        auto& p = buildP(par);
        prepare(p);
        auto phi = makeFlux(phiA);
        fvcc::BoundaryContext ctx;
        context(ctx, phi, p, 1);
        p.correctBoundaryConditions(ctx);
        const auto before = patchValues(p);

        auto phi2 = makeFlux(phiB);
        fvcc::BoundaryContext frozen;
        context(frozen, phi2, p, 1);
        frozen.insertValue(fvcc::evaluateOnlyKey, 1.0);
        p.correctBoundaryConditions(frozen);
        const auto kept = patchValues(p);
        for (std::size_t k = 0; k < 4; ++k)
            REQUIRE(kept[k] == before[k]);

        // the state did not move: the next real update equals the one of a fresh start
        // that sees phiB directly
        Reference reference(par);
        reference.update(1, dt, phiA, uPatch, pOldPatch, pNowPatch);
        const auto expected = reference.update(1, dt, phiB, uPatch, pOldPatch, before);
        fvcc::BoundaryContext next;
        context(next, phi2, p, 1);
        p.correctBoundaryConditions(next);
        const auto values = patchValues(p);
        for (std::size_t k = 0; k < 4; ++k)
            REQUIRE(values[k] == Catch::Approx(expected[k]).epsilon(1e-12));
    }

    SECTION("without a context the patch holds its initial value " + execName)
    {
        auto& p = buildP(par);
        const auto values = patchValues(p);
        for (auto v : values)
            REQUIRE(v == 100000.0);
    }

    SECTION("a context without the flux or the time index is an error " + execName)
    {
        auto& p = buildP(par);
        prepare(p);
        auto phi = makeFlux(phiA);

        fvcc::BoundaryContext noTimeIndex;
        noTimeIndex.insert("phi", phi);
        noTimeIndex.insert("U", U);
        noTimeIndex.insert("p", p);
        noTimeIndex.insertValue("deltaT", dt);
        REQUIRE_THROWS_AS(p.correctBoundaryConditions(noTimeIndex), NeoN::NeoNException);

        fvcc::BoundaryContext empty;
        REQUIRE_THROWS_AS(p.correctBoundaryConditions(empty), NeoN::NeoNException);
    }

    SECTION("attributes and entry checks " + execName)
    {
        auto bc = fvcc::VolumeBoundaryFactory<NeoN::scalar>::create(
            "plenumPressure", mesh, par.dict(), 0
        );
        REQUIRE(bc->attributes().assignable == false);
        REQUIRE(bc->attributes().fixesValue == true);

        auto dict = par.dict();
        dict.insert("rho", NeoN::scalar(1.2));
        REQUIRE_THROWS_AS(
            fvcc::VolumeBoundaryFactory<NeoN::scalar>::create("plenumPressure", mesh, dict, 0),
            NeoN::NeoNException
        );

        NeoN::Dictionary missing;
        missing.insert("type", std::string("plenumPressure"));
        REQUIRE_THROWS_AS(
            fvcc::VolumeBoundaryFactory<NeoN::scalar>::create("plenumPressure", mesh, missing, 0),
            NeoN::NeoNException
        );
    }
}
