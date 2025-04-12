// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

using Operator = NeoN::dsl::Operator;

namespace NeoN
{

template<typename ValueType>
struct CreateField
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db)
    {

        std::vector<fvcc::VolumeBoundary<ValueType>> bcs {};
        for (auto patchi : std::vector<size_t> {0, 1, 2, 3})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", ValueType(2.0));
            bcs.push_back(fvcc::VolumeBoundary<ValueType>(mesh, dict, patchi));
        }
        NeoN::DomainField<ValueType> domainField(
            mesh.exec(),
            NeoN::Field<ValueType>(mesh.exec(), mesh.nCells(), one<ValueType>()),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<ValueType> vf(mesh.exec(), name, mesh, domainField, bcs, db, "", "");

        return NeoN::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            fvcc::validateFieldDoc
        );
    }
};

TEMPLATE_TEST_CASE("DdtOperator", "[template]", NeoN::scalar, NeoN::Vector)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Database db;
    auto mesh = createSingleCellMesh(exec);
    auto sp = NeoN::finiteVolume::cellCentred::SparsityPattern {mesh};

    fvcc::FieldCollection& fieldCollection =
        fvcc::FieldCollection::instance(db, "testFieldCollection");

    fvcc::VolumeField<TestType>& phi = fieldCollection.registerField<fvcc::VolumeField<TestType>>(
        CreateField<TestType> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );
    fill(phi.internalField(), 10 * one<TestType>());
    fill(phi.boundaryField().value(), zero<TestType>());
    fill(oldTime(phi).internalField(), -1.0 * one<TestType>());
    phi.correctBoundaryConditions();

    SECTION("explicit DdtOperator " + execName)
    {
        fvcc::DdtOperator<TestType> ddtTerm(Operator::Type::Explicit, phi);
        auto source = Field<TestType>(exec, phi.size(), zero<TestType>());
        ddtTerm.explicitOperation(source, 1.0, 0.5);

        // cell has one cell
        const auto vol = mesh.cellVolumes().copyToHost();
        const auto volView = vol.view();
        auto hostSource = source.copyToHost();
        auto values = hostSource.view();
        for (auto ii = 0; ii < values.size(); ++ii)
        {
            REQUIRE(
                values[ii] == volView[0] * TestType(22.0)
            ); // => (phi^{n + 1} - phi^{n})/dt*V => (10 -- 1)/.5*V = 22V
        }
    }

    SECTION("implicit DdtOperator " + execName)
    {
        auto ls = NeoN::la::createEmptyLinearSystem<
            TestType,
            NeoN::localIdx,
            NeoN::finiteVolume::cellCentred::SparsityPattern>(sp);
        fvcc::DdtOperator<TestType> ddtTerm(Operator::Type::Implicit, phi);
        ddtTerm.implicitOperation(ls, 1.0, 0.5);

        auto lsHost = ls.copyToHost();
        const auto vol = mesh.cellVolumes().copyToHost();
        const auto volView = vol.view();
        const auto matrixValues = lsHost.matrix().values();
        const auto matrixValuesView = matrixValues.view();
        const auto rhs = lsHost.rhs().view();

        for (auto ii = 0; ii < matrixValues.size(); ++ii)
        {
            REQUIRE(
                matrixValuesView[ii] == 2.0 * volView[0] * one<TestType>()
            ); // => 1/dt*V => 1/.5*V = 2V
            REQUIRE(
                rhs[ii] == -2.0 * volView[0] * one<TestType>()
            ); // => phi^{n}/dt*V => -1/.5*V = -2V
        }
    }
}

}
