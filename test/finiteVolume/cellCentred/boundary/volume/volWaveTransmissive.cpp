// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

// Registers "p" through a VectorCollection (rather than constructing it directly), which is
// what gives it old-time storage -- needed since WaveTransmissive reads its own field's
// oldTime() boundary value. Mirrors compressiblePEqnDdt.cpp's own CreateVector helper, plus a
// patch-0 dictionary override so patch 0 can be the WaveTransmissive BC under test while
// patches 1-3 stay a plain fixedValue (their behaviour is irrelevant to these tests).
struct CreateVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    NeoN::scalar value;
    NeoN::Dictionary patch0Dict; // if it contains "type", used verbatim for patch 0

    NeoN::Document operator()(NeoN::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs {};
        for (NeoN::localIdx patchi = 0; patchi < 4; ++patchi)
        {
            if (patchi == 0 && patch0Dict.contains("type"))
            {
                bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, patch0Dict, patchi));
                continue;
            }
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", value);
            bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, dict, patchi));
        }
        NeoN::Field<NeoN::scalar> domainVector(
            mesh.exec(),
            NeoN::Vector<NeoN::scalar>(mesh.exec(), mesh.nCells(), value),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<NeoN::scalar> vf(mesh.exec(), name, mesh, domainVector, bcs, db, "", "");

        return NeoN::Document(
            {{"name", vf.name},
             {"timeIndex", std::int64_t(0)},
             {"iterationIndex", std::int64_t(0)},
             {"subCycleIndex", std::int64_t(0)},
             {"field", vf}},
            fvcc::validateVectorDoc
        );
    }
};

// Builds a uniform volume- or surface-scalar field via createCalculatedBCs, matching
// volInletOutlet.cpp's own makeFlux helper.
fvcc::VolumeField<NeoN::scalar> makeUniformVolume(
    const NeoN::Executor& exec, const NeoN::UnstructuredMesh& mesh, NeoN::scalar value
)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    fvcc::VolumeField<NeoN::scalar> field(exec, "f", mesh, bcs);
    NeoN::fill(field.internalVector(), value);
    NeoN::fill(field.boundaryData().value(), value);
    return field;
}

fvcc::SurfaceField<NeoN::scalar> makeUniformSurface(
    const NeoN::Executor& exec, const NeoN::UnstructuredMesh& mesh, NeoN::scalar value
)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceField<NeoN::scalar> field(exec, "f", mesh, bcs);
    NeoN::fill(field.internalVector(), value);
    NeoN::fill(field.boundaryData().value(), value);
    return field;
}

TEST_CASE("waveTransmissive_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Single-cell mesh, all 4 boundary faces: magSf = 1, deltaCoeffs = 2 (createSingleCellMesh).
    // Patch 0 ("left") is where WaveTransmissive is attached below.
    auto mesh = NeoN::createSingleCellMesh(exec);

    const NeoN::scalar gamma = 1.4;
    const NeoN::scalar dt = 0.5;
    const NeoN::scalar pOldValue = 100000.0;
    const NeoN::scalar pInternalValue = 100010.0;

    // db lives for the rest of this TEST_CASE invocation (Catch2 re-enters the whole function
    // from scratch for every SECTION leaf run, so this is a fresh Database each time, not
    // shared/stale across sections) -- registerVector's returned reference must not outlive
    // it, so it is declared here rather than inside buildP.
    NeoN::Database db;
    auto buildP = [&](const NeoN::Dictionary& patch0Dict) -> fvcc::VolumeField<NeoN::scalar>&
    {
        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(db, "testVectorCollection");
        fvcc::VolumeField<NeoN::scalar>& p =
            vectorCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(CreateVector {
                .name = "p", .mesh = mesh, .value = pInternalValue, .patch0Dict = patch0Dict
            });
        p.correctBoundaryConditions();
        NeoN::fill(oldTime(p).internalVector(), pOldValue);
        NeoN::fill(oldTime(p).boundaryData().value(), pOldValue);
        return p;
    };

    // Patch 0's range, by createSingleCellMesh's own fixed offsets {0,1,2,3,4}.
    const std::pair<NeoN::localIdx, NeoN::localIdx> patch0Range {0, 1};

    SECTION("outflow, no far-field relaxation (lInf unset) " + execName)
    {
        // Un = (phi/rho)/magSf = 2, c = sqrt(gamma/psi) = sqrt(1.4), w = Un + c (both positive,
        // no clamping), alpha = w*dt*deltaCoeffs. Hand-computed (see the feature commit's own
        // message for the exact derivation):
        //   w       = 3.1832159566199234
        //   alpha   = 3.1832159566199234
        //   refValue (= p_oldTime, lInf disabled) = 100000.0
        //   valueFraction = 1/(1+alpha)            = 0.2390505320236943
        //   value   = valueFraction*refValue + (1-valueFraction)*p_internal = 100007.60949467975
        auto phi = makeUniformSurface(exec, mesh, 2.0);
        auto rho = makeUniformVolume(exec, mesh, 1.0);
        auto psi = makeUniformVolume(exec, mesh, 1.0);

        NeoN::Dictionary patch0Dict;
        patch0Dict.insert("type", std::string("waveTransmissive"));
        patch0Dict.insert("gamma", gamma);
        auto& p = buildP(patch0Dict);

        REQUIRE(
            fvcc::VolumeBoundaryFactory<NeoN::scalar>::create(
                "waveTransmissive", mesh, patch0Dict, 0
            )
                ->attributes()
                .fixesValue
            == false
        );

        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);
        ctx.insert("rho", rho);
        ctx.insert("psi", psi);
        ctx.insert("p", p);
        ctx.insertValue("deltaT", dt);

        p.correctBoundaryConditions(ctx);

        auto [valuesH, refValuesH, refGradH, valueFractionH] = copyToHosts(
            p.boundaryData().value(),
            p.boundaryData().refValue(),
            p.boundaryData().refGrad(),
            p.boundaryData().valueFraction()
        );

        for (auto& v : refValuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(100000.0));
        for (auto& f : valueFractionH.view(patch0Range))
            REQUIRE(f == Catch::Approx(0.2390505320236943));
        for (auto& g : refGradH.view(patch0Range))
            REQUIRE(g == Catch::Approx(0.0));
        for (auto& v : valuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(100007.60949467975));
    }

    SECTION("outflow, with far-field relaxation (lInf set) " + execName)
    {
        // Same w/alpha as above; lInf=2, fieldInf=90000:
        //   k             = w*dt/lInf          = 0.7958039891549809
        //   refValue      = (p_old+k*fieldInf)/(1+k) = 95568.53646633534
        //   valueFraction = (1+k)/(1+alpha+k)  = 0.3606741906464672
        //   value                              = 98408.0787347097
        auto phi = makeUniformSurface(exec, mesh, 2.0);
        auto rho = makeUniformVolume(exec, mesh, 1.0);
        auto psi = makeUniformVolume(exec, mesh, 1.0);

        NeoN::Dictionary patch0Dict;
        patch0Dict.insert("type", std::string("waveTransmissive"));
        patch0Dict.insert("gamma", gamma);
        patch0Dict.insert("fieldInf", NeoN::scalar(90000.0));
        patch0Dict.insert("lInf", NeoN::scalar(2.0));
        auto& p = buildP(patch0Dict);

        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);
        ctx.insert("rho", rho);
        ctx.insert("psi", psi);
        ctx.insert("p", p);
        ctx.insertValue("deltaT", dt);

        p.correctBoundaryConditions(ctx);

        auto [valuesH, refValuesH, valueFractionH] = copyToHosts(
            p.boundaryData().value(), p.boundaryData().refValue(), p.boundaryData().valueFraction()
        );

        for (auto& v : refValuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(95568.53646633534));
        for (auto& f : valueFractionH.view(patch0Range))
            REQUIRE(f == Catch::Approx(0.3606741906464672));
        for (auto& v : valuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(98408.0787347097));
    }

    SECTION(
        "strong inflow clamps the wave speed to zero -> pure Dirichlet at p_oldTime " + execName
    )
    {
        // Un = -5 (phi = -5, rho = 1, magSf = 1), c = sqrt(1.4) ~ 1.183: Un + c < 0, clamped to
        // w = 0 -> alpha = 0 -> valueFraction = 1 -> value = refValue = p_oldTime exactly,
        // independent of p_internal.
        auto phi = makeUniformSurface(exec, mesh, -5.0);
        auto rho = makeUniformVolume(exec, mesh, 1.0);
        auto psi = makeUniformVolume(exec, mesh, 1.0);

        NeoN::Dictionary patch0Dict;
        patch0Dict.insert("type", std::string("waveTransmissive"));
        patch0Dict.insert("gamma", gamma);
        auto& p = buildP(patch0Dict);

        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);
        ctx.insert("rho", rho);
        ctx.insert("psi", psi);
        ctx.insert("p", p);
        ctx.insertValue("deltaT", dt);

        p.correctBoundaryConditions(ctx);

        auto [valuesH, valueFractionH] =
            copyToHosts(p.boundaryData().value(), p.boundaryData().valueFraction());

        for (auto& f : valueFractionH.view(patch0Range))
            REQUIRE(f == Catch::Approx(1.0));
        for (auto& v : valuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(pOldValue));
    }

    SECTION("missing context piece => degenerates to zero-gradient " + execName)
    {
        auto phi = makeUniformSurface(exec, mesh, 2.0);
        auto rho = makeUniformVolume(exec, mesh, 1.0);
        // psi deliberately omitted from the context below.

        NeoN::Dictionary patch0Dict;
        patch0Dict.insert("type", std::string("waveTransmissive"));
        patch0Dict.insert("gamma", gamma);
        auto& p = buildP(patch0Dict);

        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);
        ctx.insert("rho", rho);
        ctx.insert("p", p);
        ctx.insertValue("deltaT", dt);

        p.correctBoundaryConditions(ctx);

        auto [valuesH, valueFractionH] =
            copyToHosts(p.boundaryData().value(), p.boundaryData().valueFraction());

        for (auto& f : valueFractionH.view(patch0Range))
            REQUIRE(f == Catch::Approx(0.0));
        for (auto& v : valuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(pInternalValue));
    }

    SECTION("no context at all => degenerates to zero-gradient " + execName)
    {
        NeoN::Dictionary patch0Dict;
        patch0Dict.insert("type", std::string("waveTransmissive"));
        patch0Dict.insert("gamma", gamma);
        auto& p = buildP(patch0Dict);

        p.correctBoundaryConditions();

        auto [valuesH, valueFractionH] =
            copyToHosts(p.boundaryData().value(), p.boundaryData().valueFraction());

        for (auto& f : valueFractionH.view(patch0Range))
            REQUIRE(f == Catch::Approx(0.0));
        for (auto& v : valuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(pInternalValue));
    }
}
