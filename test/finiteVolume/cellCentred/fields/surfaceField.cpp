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
    SECTION("arithmetic rejects operands of different sizes on: " + execName)
    {
        // 1D meshes of 4 and 5 cells: 3 and 4 internal faces, both with 2 boundary faces
        NeoN::UnstructuredMesh mesh4 = NeoN::create1DUniformMesh(exec, 4);
        NeoN::UnstructuredMesh mesh5 = NeoN::create1DUniformMesh(exec, 5);
        auto sBcs4 = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh4);
        auto sBcs5 = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh5);
        auto vBcs4 = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::Vec3>>(mesh4);
        auto vBcs5 = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::Vec3>>(mesh5);

        fvcc::SurfaceField<NeoN::scalar> s4(exec, "s4", mesh4, sBcs4);
        fvcc::SurfaceField<NeoN::scalar> s5(exec, "s5", mesh5, sBcs5);
        fvcc::SurfaceField<NeoN::Vec3> v4(exec, "v4", mesh4, vBcs4);
        fvcc::SurfaceField<NeoN::Vec3> v5(exec, "v5", mesh5, vBcs5);
        NeoN::fill(s4.internalVector(), 1.0);
        NeoN::fill(s5.internalVector(), 1.0);
        NeoN::fill(v4.internalVector(), NeoN::Vec3(1.0, 1.0, 1.0));
        NeoN::fill(v5.internalVector(), NeoN::Vec3(1.0, 1.0, 1.0));

        REQUIRE_THROWS(s4 * s5);
        REQUIRE_THROWS(s4 + s5);
        REQUIRE_THROWS(s4 - s5);
        REQUIRE_THROWS(s4 * v5);
        REQUIRE_THROWS(s5 * v4);
        REQUIRE_THROWS(v4 * s5);
        REQUIRE_THROWS(v4 + v5);
        REQUIRE_THROWS(v5 - v4);

        // operands of equal size still work
        REQUIRE_NOTHROW(s4 * s4);
        REQUIRE_NOTHROW(s4 * v4);
        REQUIRE_NOTHROW(v4 - v4);
    }

    SECTION("arithmetic rejects boundary data of a different size on: " + execName)
    {
        // single-cell mesh: 0 internal and 4 boundary faces; 1D mesh of 4 cells: 3 and 2
        NeoN::UnstructuredMesh single = NeoN::createSingleCellMesh(exec);
        NeoN::UnstructuredMesh line = NeoN::create1DUniformMesh(exec, 4);
        fvcc::SurfaceField<NeoN::scalar> a(
            exec,
            "a",
            single,
            fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(single)
        );
        fvcc::SurfaceField<NeoN::scalar> b(
            exec, "b", line, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(line)
        );
        REQUIRE_THROWS(a + b);
    }

    SECTION("arithmetic rejects operands on different executors on: " + execName)
    {
        NeoN::Executor serial = NeoN::SerialExecutor {};
        if (exec != serial)
        {
            NeoN::UnstructuredMesh meshSerial = NeoN::create1DUniformMesh(serial, 4);
            NeoN::UnstructuredMesh meshExec = NeoN::create1DUniformMesh(exec, 4);
            fvcc::SurfaceField<NeoN::scalar> a(
                serial,
                "a",
                meshSerial,
                fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(meshSerial)
            );
            fvcc::SurfaceField<NeoN::scalar> b(
                exec,
                "b",
                meshExec,
                fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(meshExec)
            );
            REQUIRE_THROWS(a + b);
        }
    }

    SECTION("an arithmetic result keeps its computed boundary values on: " + execName)
    {
        // The operands carry fixedValue boundary conditions. A result that copied them would
        // reset its boundary values to the operand's stored value at the next
        // correctBoundaryConditions(), although the values were just computed.
        NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);
        auto fixed = [&](auto value)
        {
            using V = decltype(value);
            std::vector<fvcc::SurfaceBoundary<V>> bcs {};
            for (auto patchi : I<NeoN::localIdx> {0, 1, 2, 3})
            {
                NeoN::Dictionary dict;
                dict.insert("type", std::string("fixedValue"));
                dict.insert("fixedValue", value);
                bcs.push_back(fvcc::SurfaceBoundary<V>(mesh, dict, patchi));
            }
            return bcs;
        };
        fvcc::SurfaceField<NeoN::scalar> s(exec, "s", mesh, fixed(2.0));
        fvcc::SurfaceField<NeoN::scalar> t(exec, "t", mesh, fixed(3.0));
        fvcc::SurfaceField<NeoN::Vec3> a(exec, "a", mesh, fixed(NeoN::Vec3(1.0, 2.0, 3.0)));
        fvcc::SurfaceField<NeoN::Vec3> b(exec, "b", mesh, fixed(NeoN::Vec3(0.5, 0.5, 0.5)));
        for (auto* f : {&s, &t})
            f->correctBoundaryConditions();
        for (auto* f : {&a, &b})
            f->correctBoundaryConditions();

        auto requireBoundary = [](const auto& field, auto expected)
        {
            auto host = field.boundaryData().value().copyToHost();
            REQUIRE(host.size() == 4);
            for (NeoN::localIdx i = 0; i < host.size(); ++i)
            {
                REQUIRE(host.view()[i] == expected);
            }
        };

        auto st = s * t;
        auto sum = s + t;
        auto diff = t - s;
        auto scaled = s * a;
        auto vsum = a + b;
        auto vdiff = a - b;
        auto scaledByNumber = 4.0 * s;
        for (auto* f : {&st, &sum, &diff, &scaledByNumber})
            f->correctBoundaryConditions();
        for (auto* f : {&scaled, &vsum, &vdiff})
            f->correctBoundaryConditions();

        requireBoundary(st, 6.0);
        requireBoundary(sum, 5.0);
        requireBoundary(diff, 1.0);
        requireBoundary(scaledByNumber, 8.0);
        requireBoundary(scaled, NeoN::Vec3(2.0, 4.0, 6.0));
        requireBoundary(vsum, NeoN::Vec3(1.5, 2.5, 3.5));
        requireBoundary(vdiff, NeoN::Vec3(0.5, 1.5, 2.5));

        // the operands are not changed by building or correcting a result
        requireBoundary(s, 2.0);
        requireBoundary(a, NeoN::Vec3(1.0, 2.0, 3.0));
    }
}
