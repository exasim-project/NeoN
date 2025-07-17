// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

using Operator = NeoN::dsl::Operator;

namespace NeoN
{

template<typename ValueType>
struct CreateVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<ValueType>> bcs {};
        for (auto patchi : std::vector<NeoN::localIdx> {0, 1, 2, 3})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", ValueType(2.0));
            bcs.push_back(fvcc::VolumeBoundary<ValueType>(mesh, dict, patchi));
        }
        NeoN::Field<ValueType> domainVector(
            mesh.exec(),
            NeoN::Vector<ValueType>(mesh.exec(), mesh.nCells(), one<ValueType>()),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<ValueType> vf(mesh.exec(), name, mesh, domainVector, bcs, db, "", "");

        return NeoN::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            fvcc::validateVectorDoc
        );
    }
};

TEMPLATE_TEST_CASE("DdtOperator", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Database db;
    auto mesh = createSingleCellMesh(exec);
    auto sp = NeoN::la::SparsityPattern {mesh};

    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "testVectorCollection");

    fvcc::VolumeField<TestType>& phi = fieldCollection.registerVector<fvcc::VolumeField<TestType>>(
        CreateVector<TestType> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );
    fill(phi.internalVector(), 10 * one<TestType>());
    fill(phi.boundaryData().value(), zero<TestType>());
    fill(oldTime(phi).internalVector(), -1.0 * one<TestType>());
    phi.correctBoundaryConditions();

    SECTION("explicit DdtOperator " + execName)
    {
        auto ddtOp = dsl::exp::ddt(phi);
        auto source = Vector<TestType>(exec, phi.size(), zero<TestType>());
        ddtOp.explicitOperation(source, 1.0, 0.5);

        const auto [vol, hostSource] = copyToHosts(mesh.cellVolumes(), source);
        const auto [volV, vals] = views(vol, hostSource);

        for (auto ii = 0; ii < vals.size(); ++ii)
        {
            // => (phi^{n + 1} - phi^{n})/dt*V => (10 -- 1)/.5*V = 22V
            REQUIRE(vals[ii] == volV[0] * TestType(22.0));
        }
    }

    SECTION("implicit DdtOperator " + execName)
    {
        auto ls = NeoN::la::createEmptyLinearSystem<TestType, NeoN::localIdx>(mesh, sp);

        auto ddtOp = dsl::imp::ddt(phi);
        ddtOp.implicitOperation(ls, 1.0, 0.5);

        const auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
        const auto [mtxValsV, volV, rhsV] = views(lsHost.matrix().values(), vol, lsHost.rhs());

        for (auto ii = 0; ii < mtxValsV.size(); ++ii)
        {
            // => 1/dt*V => 1/.5*V = 2V
            REQUIRE(mtxValsV[ii] == 2.0 * volV[0] * one<TestType>());
            // => phi^{n}/dt*V => -1/.5*V = -2V
            REQUIRE(rhsV[ii] == -2.0 * volV[0] * one<TestType>());
        }
    }
}

}
