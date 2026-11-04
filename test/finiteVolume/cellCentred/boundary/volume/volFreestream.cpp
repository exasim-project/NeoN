// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include <cmath>

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace
{

// Unit face normals of the boundary faces of `mesh`, on the host.
std::vector<NeoN::Vec3> unitNormals(const NeoN::UnstructuredMesh& mesh)
{
    auto sfH = mesh.boundaryMesh().faceNormals().copyToHost();
    std::vector<NeoN::Vec3> nf;
    for (NeoN::localIdx i = 0; i < static_cast<NeoN::localIdx>(sfH.size()); ++i)
    {
        const NeoN::Vec3 sf = sfH.view()[i];
        const NeoN::scalar mag = std::sqrt(sf[0] * sf[0] + sf[1] * sf[1] + sf[2] * sf[2]);
        nf.push_back(NeoN::Vec3(sf[0] / mag, sf[1] / mag, sf[2] / mag));
    }
    return nf;
}

NeoN::scalar dotProduct(const NeoN::Vec3& a, const NeoN::Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

NeoN::scalar magnitude(const NeoN::Vec3& a) { return std::sqrt(dotProduct(a, a)); }

// createSingleCellMesh has four boundary PATCHES (left, top, right, bottom) with one face each,
// so every test creates one boundary object per patch and applies them all.
constexpr NeoN::localIdx nPatches = 4;

template<typename ValueType>
std::vector<std::unique_ptr<fvcc::VolumeBoundaryFactory<ValueType>>> createPatches(
    const std::string& type, const NeoN::UnstructuredMesh& mesh, const NeoN::Dictionary& dict
)
{
    std::vector<std::unique_ptr<fvcc::VolumeBoundaryFactory<ValueType>>> patches;
    for (NeoN::localIdx patchID = 0; patchID < nPatches; ++patchID)
    {
        patches.push_back(fvcc::VolumeBoundaryFactory<ValueType>::create(type, mesh, dict, patchID)
        );
    }
    return patches;
}

} // namespace

TEST_CASE("freestreamVelocity_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = NeoN::createSingleCellMesh(exec);
    const auto nf = unitNormals(mesh);
    const NeoN::Vec3 freestream(200.0, 0.0, 0.0);
    const NeoN::Vec3 cellValue(-1.0, -2.0, -3.0);

    NeoN::Dictionary dict;
    dict.insert("type", std::string("freestreamVelocity"));
    dict.insert("freestreamValue", freestream);
    auto patches = createPatches<NeoN::Vec3>("freestreamVelocity", mesh, dict);
    auto& boundary = patches[0];

    auto run = [&](const NeoN::Vec3& patchValue)
    {
        auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), cellValue);
        NeoN::fill(field.boundaryData().value(), patchValue);
        for (auto& patch : patches)
            patch->correctBoundaryCondition(field);
        return field;
    };

    SECTION("attributes: a mixed, assignable, non-Dirichlet patch " + execName)
    {
        REQUIRE(boundary->attributes().fixesValue == false);
        REQUIRE(boundary->attributes().mixed == true);
        REQUIRE(boundary->attributes().assignable == true);
    }

    SECTION("valueFraction = 0.5 - 0.5 (U.nf)/|U| on every face " + execName)
    {
        for (const NeoN::Vec3 patchValue :
             {NeoN::Vec3(2.0, 0.0, 0.0), NeoN::Vec3(3.0, 4.0, 0.0), NeoN::Vec3(-1.0, 2.0, 0.5)})
        {
            auto field = run(patchValue);
            auto [valuesH, refValuesH, refGradH, valueFractionH] = copyToHosts(
                field.boundaryData().value(),
                field.boundaryData().refValue(),
                field.boundaryData().refGrad(),
                field.boundaryData().valueFraction()
            );
            const std::pair<NeoN::localIdx, NeoN::localIdx> range {0, nPatches};
            for (NeoN::localIdx i = range.first; i < range.second; ++i)
            {
                const NeoN::scalar expected =
                    0.5
                    - 0.5 * dotProduct(patchValue, nf[static_cast<std::size_t>(i)])
                          / magnitude(patchValue);
                REQUIRE(valueFractionH.view()[i] == Catch::Approx(expected));
                for (int c = 0; c < 3; ++c)
                {
                    REQUIRE(refValuesH.view()[i][c] == Catch::Approx(freestream[c]));
                    REQUIRE(refGradH.view()[i][c] == Catch::Approx(0.0));
                    REQUIRE(
                        valuesH.view()[i][c]
                        == Catch::Approx(freestream[c] * expected + cellValue[c] * (1.0 - expected))
                    );
                }
            }
        }
    }

    SECTION("pure outflow is zero-gradient, pure inflow takes the freestream value " + execName)
    {
        // U along +x: the +x face is outflow (valueFraction 0), the -x face inflow (1)
        auto field = run(NeoN::Vec3(2.0, 0.0, 0.0));
        auto [valuesH, valueFractionH] =
            copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());
        bool sawOutflow = false;
        bool sawInflow = false;
        const std::pair<NeoN::localIdx, NeoN::localIdx> range {0, nPatches};
        for (NeoN::localIdx i = range.first; i < range.second; ++i)
        {
            const NeoN::Vec3 n = nf[static_cast<std::size_t>(i)];
            if (n[0] > 0.9)
            {
                sawOutflow = true;
                REQUIRE(valueFractionH.view()[i] == Catch::Approx(0.0));
                REQUIRE(valuesH.view()[i][0] == Catch::Approx(cellValue[0]));
            }
            if (n[0] < -0.9)
            {
                sawInflow = true;
                REQUIRE(valueFractionH.view()[i] == Catch::Approx(1.0));
                REQUIRE(valuesH.view()[i][0] == Catch::Approx(freestream[0]));
            }
        }
        REQUIRE(sawOutflow);
        REQUIRE(sawInflow);
    }

    SECTION("a zero patch value falls back to the freestream direction " + execName)
    {
        auto field = run(NeoN::Vec3(0.0, 0.0, 0.0));
        auto valueFractionH = field.boundaryData().valueFraction().copyToHost();
        const std::pair<NeoN::localIdx, NeoN::localIdx> range {0, nPatches};
        for (NeoN::localIdx i = range.first; i < range.second; ++i)
        {
            const NeoN::scalar expected =
                0.5
                - 0.5 * dotProduct(freestream, nf[static_cast<std::size_t>(i)])
                      / magnitude(freestream);
            REQUIRE(std::isfinite(valueFractionH.view()[i]));
            REQUIRE(valueFractionH.view()[i] == Catch::Approx(expected));
        }
    }

    SECTION("the context-aware overload gives the same result " + execName)
    {
        auto plain = run(NeoN::Vec3(3.0, 4.0, 0.0));
        auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), cellValue);
        NeoN::fill(field.boundaryData().value(), NeoN::Vec3(3.0, 4.0, 0.0));
        fvcc::BoundaryContext ctx;
        for (auto& patch : patches)
            patch->correctBoundaryCondition(field, ctx);
        auto aH = plain.boundaryData().valueFraction().copyToHost();
        auto bH = field.boundaryData().valueFraction().copyToHost();
        const std::pair<NeoN::localIdx, NeoN::localIdx> range {0, nPatches};
        for (NeoN::localIdx i = range.first; i < range.second; ++i)
        {
            REQUIRE(aH.view()[i] == Catch::Approx(bH.view()[i]));
        }
    }
}

TEST_CASE("freestreamPressure_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = NeoN::createSingleCellMesh(exec);
    const auto nf = unitNormals(mesh);
    const NeoN::scalar freestream = 1.0e5;
    const NeoN::scalar cellValue = 9.0e4;

    auto makeBoundary = [&](const std::string& uName)
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("freestreamPressure"));
        dict.insert("freestreamValue", freestream);
        if (!uName.empty()) dict.insert("U", uName);
        return createPatches<NeoN::scalar>("freestreamPressure", mesh, dict);
    };
    auto makeU = [&](const std::string& name, const NeoN::Vec3& patchValue)
    {
        fvcc::VolumeField<NeoN::Vec3> U(
            exec, name, mesh, fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh)
        );
        NeoN::fill(U.internalVector(), patchValue);
        NeoN::fill(U.boundaryData().value(), patchValue);
        return U;
    };
    auto makeP = [&]()
    {
        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), cellValue);
        return field;
    };

    auto patches = makeBoundary("");
    auto& boundary = patches[0];

    SECTION("attributes: a mixed, assignable, non-Dirichlet patch " + execName)
    {
        REQUIRE(boundary->attributes().fixesValue == false);
        REQUIRE(boundary->attributes().mixed == true);
        REQUIRE(boundary->attributes().assignable == true);
    }

    SECTION("valueFraction = 0.5 + 0.5 (U.nf)/|U| from the velocity in the context " + execName)
    {
        for (const NeoN::Vec3 patchValue :
             {NeoN::Vec3(2.0, 0.0, 0.0), NeoN::Vec3(3.0, 4.0, 0.0), NeoN::Vec3(-1.0, 2.0, 0.5)})
        {
            auto U = makeU("U", patchValue);
            fvcc::BoundaryContext ctx;
            ctx.insert("U", U);
            auto field = makeP();
            for (auto& patch : patches)
                patch->correctBoundaryCondition(field, ctx);

            auto [valuesH, refValuesH, refGradH, valueFractionH] = copyToHosts(
                field.boundaryData().value(),
                field.boundaryData().refValue(),
                field.boundaryData().refGrad(),
                field.boundaryData().valueFraction()
            );
            const std::pair<NeoN::localIdx, NeoN::localIdx> range {0, nPatches};
            for (NeoN::localIdx i = range.first; i < range.second; ++i)
            {
                const NeoN::scalar expected =
                    0.5
                    + 0.5 * dotProduct(patchValue, nf[static_cast<std::size_t>(i)])
                          / magnitude(patchValue);
                REQUIRE(valueFractionH.view()[i] == Catch::Approx(expected));
                REQUIRE(refValuesH.view()[i] == Catch::Approx(freestream));
                REQUIRE(refGradH.view()[i] == Catch::Approx(0.0));
                REQUIRE(
                    valuesH.view()[i]
                    == Catch::Approx(freestream * expected + cellValue * (1.0 - expected))
                );
            }
        }
    }

    SECTION("outflow fixes the freestream pressure, inflow is zero-gradient " + execName)
    {
        auto U = makeU("U", NeoN::Vec3(2.0, 0.0, 0.0));
        fvcc::BoundaryContext ctx;
        ctx.insert("U", U);
        auto field = makeP();
        for (auto& patch : patches)
            patch->correctBoundaryCondition(field, ctx);
        auto [valuesH, valueFractionH] =
            copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());
        bool sawOutflow = false;
        bool sawInflow = false;
        const std::pair<NeoN::localIdx, NeoN::localIdx> range {0, nPatches};
        for (NeoN::localIdx i = range.first; i < range.second; ++i)
        {
            const NeoN::Vec3 n = nf[static_cast<std::size_t>(i)];
            if (n[0] > 0.9)
            {
                sawOutflow = true;
                REQUIRE(valueFractionH.view()[i] == Catch::Approx(1.0));
                REQUIRE(valuesH.view()[i] == Catch::Approx(freestream));
            }
            if (n[0] < -0.9)
            {
                sawInflow = true;
                REQUIRE(valueFractionH.view()[i] == Catch::Approx(0.0));
                REQUIRE(valuesH.view()[i] == Catch::Approx(cellValue));
            }
        }
        REQUIRE(sawOutflow);
        REQUIRE(sawInflow);
    }

    SECTION("without a velocity it degenerates to zero-gradient " + execName)
    {
        auto field = makeP();
        for (auto& patch : patches)
            patch->correctBoundaryCondition(field);
        auto [valuesH, valueFractionH] =
            copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());
        const std::pair<NeoN::localIdx, NeoN::localIdx> range {0, nPatches};
        for (NeoN::localIdx i = range.first; i < range.second; ++i)
        {
            REQUIRE(valueFractionH.view()[i] == Catch::Approx(0.0));
            REQUIRE(valuesH.view()[i] == Catch::Approx(cellValue));
        }

        // a context that lacks the velocity behaves the same
        auto field2 = makeP();
        fvcc::BoundaryContext ctx;
        for (auto& patch : patches)
            patch->correctBoundaryCondition(field2, ctx);
        auto valueFraction2H = field2.boundaryData().valueFraction().copyToHost();
        for (NeoN::localIdx i = range.first; i < range.second; ++i)
        {
            REQUIRE(valueFraction2H.view()[i] == Catch::Approx(0.0));
        }
    }

    SECTION("the velocity field name is configurable " + execName)
    {
        auto namedPatches = makeBoundary("Uinf");
        auto U = makeU("Uinf", NeoN::Vec3(2.0, 0.0, 0.0));
        fvcc::BoundaryContext ctx;
        ctx.insert("Uinf", U);
        auto field = makeP();
        for (auto& patch : namedPatches)
            patch->correctBoundaryCondition(field, ctx);
        auto valueFractionH = field.boundaryData().valueFraction().copyToHost();
        const std::pair<NeoN::localIdx, NeoN::localIdx> range {0, nPatches};
        for (NeoN::localIdx i = range.first; i < range.second; ++i)
        {
            const NeoN::scalar expected = 0.5 + 0.5 * nf[static_cast<std::size_t>(i)][0];
            REQUIRE(valueFractionH.view()[i] == Catch::Approx(expected));
        }
    }
}
