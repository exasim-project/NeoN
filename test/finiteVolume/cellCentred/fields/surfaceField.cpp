// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("surfaceVector")
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("can instantiate SurfaceField with fixedValues on: " + execName)
    {
        NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);
        std::vector<fvcc::SurfaceBoundary<NeoN::scalar>> bcs {};
        for (auto patchi : I<NeoN::localIdx> {0, 1, 2, 3})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 2.0);
            bcs.push_back(fvcc::SurfaceBoundary<NeoN::scalar>(mesh, dict, patchi));
        }

        fvcc::SurfaceField<NeoN::scalar> sf(exec, "sf", mesh, bcs);
        // single-cell mesh has 0 internal faces and 4 boundary faces
        REQUIRE(sf.internalVector().size() == 0);
        REQUIRE(sf.boundaryData().value().size() == 4);

        // Fill boundary values with 1.0 before correctBoundaryConditions
        NeoN::fill(sf.boundaryData().value(), 1.0);
        {
            auto bValues = sf.boundaryData().value().copyToHost();
            for (NeoN::localIdx i = 0; i < bValues.size(); ++i)
            {
                REQUIRE(bValues.view()[i] == 1.0);
            }
        }

        sf.correctBoundaryConditions();
        REQUIRE(sf.exec() == exec);

        // correctBoundaryConditions sets boundary data to fixedValue (2.0)
        auto values = sf.boundaryData().value().copyToHost();
        for (NeoN::localIdx i = 0; i < values.size(); ++i)
        {
            REQUIRE(values.view()[i] == 2.0);
        }

        auto refValue = sf.boundaryData().refValue().copyToHost();
        for (NeoN::localIdx i = 0; i < refValue.size(); ++i)
        {
            REQUIRE(refValue.view()[i] == 2.0);
        }
    }

    SECTION("scalar * Vec3 surface field arithmetic on: " + execName)
    {
        NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);

        std::vector<fvcc::SurfaceBoundary<NeoN::scalar>> scalarBcs {};
        std::vector<fvcc::SurfaceBoundary<NeoN::Vec3>> vectorBcs {};
        for (auto patchi : I<NeoN::localIdx> {0, 1, 2, 3})
        {
            NeoN::Dictionary scalarDict;
            scalarDict.insert("type", std::string("fixedValue"));
            scalarDict.insert("fixedValue", 2.0);
            scalarBcs.push_back(fvcc::SurfaceBoundary<NeoN::scalar>(mesh, scalarDict, patchi));

            NeoN::Dictionary vectorDict;
            vectorDict.insert("type", std::string("fixedValue"));
            vectorDict.insert("fixedValue", NeoN::Vec3(1.0, 2.0, 3.0));
            vectorBcs.push_back(fvcc::SurfaceBoundary<NeoN::Vec3>(mesh, vectorDict, patchi));
        }

        fvcc::SurfaceField<NeoN::scalar> rho(exec, "rho", mesh, scalarBcs);
        rho.correctBoundaryConditions();

        fvcc::SurfaceField<NeoN::Vec3> uf(exec, "uf", mesh, vectorBcs);
        uf.correctBoundaryConditions();

        auto rhoUf = rho * uf;
        auto hostValues = rhoUf.boundaryData().value().copyToHost();
        for (NeoN::localIdx i = 0; i < hostValues.size(); ++i)
        {
            // rho=2.0, uf=(1,2,3) => rhoUf = (2,4,6)
            REQUIRE(hostValues.view()[i] == NeoN::Vec3(2.0, 4.0, 6.0));
        }

        auto ufRho = uf * rho;
        auto hostValues2 = ufRho.boundaryData().value().copyToHost();
        for (NeoN::localIdx i = 0; i < hostValues2.size(); ++i)
        {
            REQUIRE(hostValues2.view()[i] == NeoN::Vec3(2.0, 4.0, 6.0));
        }
    }

    SECTION("Vec3 surface field arithmetic with internal faces on: " + execName)
    {
        // The single-cell mesh above has no internal faces, so it only exercises the
        // boundary values. A 1D mesh of 4 cells has 3 internal and 2 boundary faces.
        NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, 4);
        auto scalarBcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
        auto vectorBcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::Vec3>>(mesh);

        fvcc::SurfaceField<NeoN::scalar> s(exec, "s", mesh, scalarBcs);
        fvcc::SurfaceField<NeoN::Vec3> a(exec, "a", mesh, vectorBcs);
        fvcc::SurfaceField<NeoN::Vec3> b(exec, "b", mesh, vectorBcs);

        // distinct internal and boundary values, so a field that only handles one of the
        // two (or mixes them up) fails
        NeoN::fill(s.internalVector(), 2.0);
        NeoN::fill(s.boundaryData().value(), 3.0);
        NeoN::fill(a.internalVector(), NeoN::Vec3(1.0, 2.0, 3.0));
        NeoN::fill(a.boundaryData().value(), NeoN::Vec3(4.0, 5.0, 6.0));
        NeoN::fill(b.internalVector(), NeoN::Vec3(0.5, 0.5, 0.5));
        NeoN::fill(b.boundaryData().value(), NeoN::Vec3(1.0, 1.0, 1.0));

        REQUIRE(a.internalVector().size() == 3);
        REQUIRE(a.boundaryData().value().size() == 2);

        auto requireAll = [](const NeoN::Vector<NeoN::Vec3>& field, NeoN::Vec3 expected)
        {
            auto host = field.copyToHost();
            REQUIRE(host.size() > 0);
            for (NeoN::localIdx i = 0; i < host.size(); ++i)
            {
                REQUIRE(host.view()[i] == expected);
            }
        };

        // scalar * Vec3, in both orders
        for (const auto& scaled : {s * a, a * s})
        {
            requireAll(scaled.internalVector(), NeoN::Vec3(2.0, 4.0, 6.0));
            requireAll(scaled.boundaryData().value(), NeoN::Vec3(12.0, 15.0, 18.0));
        }

        auto sum = a + b;
        requireAll(sum.internalVector(), NeoN::Vec3(1.5, 2.5, 3.5));
        requireAll(sum.boundaryData().value(), NeoN::Vec3(5.0, 6.0, 7.0));

        auto difference = a - b;
        requireAll(difference.internalVector(), NeoN::Vec3(0.5, 1.5, 2.5));
        requireAll(difference.boundaryData().value(), NeoN::Vec3(3.0, 4.0, 5.0));

        // the operands are left untouched
        requireAll(a.internalVector(), NeoN::Vec3(1.0, 2.0, 3.0));
        requireAll(a.boundaryData().value(), NeoN::Vec3(4.0, 5.0, 6.0));
    }
}
