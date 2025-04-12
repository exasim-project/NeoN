// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"
#include <string>

#include "../dsl/common.hpp"

#include "NeoN/NeoN.hpp"


namespace fvcc = NeoN::finiteVolume::cellCentred;

using Field = NeoN::Field<NeoN::scalar>;
using Coeff = NeoN::dsl::Coeff;
using SpatialOperator = NeoN::dsl::SpatialOperator<NeoN::scalar>;
using TemporalOperator = NeoN::dsl::TemporalOperator<NeoN::scalar>;
using Executor = NeoN::Executor;
using VolumeField = fvcc::VolumeField<NeoN::scalar>;
using OperatorMixin = NeoN::dsl::OperatorMixin<VolumeField>;
using BoundaryData = NeoN::BoundaryData<NeoN::scalar>;
using Ddt = NeoN::dsl::temporal::Ddt<VolumeField>;

// only for msvc
template class NeoN::timeIntegration::RungeKutta<VolumeField>;

class YSquared : public OperatorMixin
{

public:

    using FieldValueType = NeoN::scalar;

    YSquared(VolumeField& field)
        : OperatorMixin(field.exec(), dsl::Coeff(1.0), field, Operator::Type::Explicit)
    {}

    void explicitOperation(Field& source) const
    {
        auto sourceView = source.view();
        auto fieldData = field_.internalField().data();
        NeoN::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) { sourceView[i] -= fieldData[i] * fieldData[i]; }
        );
    }

    std::string getName() const { return "YSquared"; }
};

struct CreateField
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs {};
        NeoN::DomainField<NeoN::scalar> domainField(
            mesh.exec(),
            NeoN::Field<NeoN::scalar>(mesh.exec(), mesh.nCells(), 1.0),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<NeoN::scalar> vf(mesh.exec(), name, mesh, domainField, bcs, db, "", "");
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

TEST_CASE("TimeIntegration - Runge Kutta")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    NeoN::scalar convergenceTolerance = 1.0e-4; // how much lower we accept that expected order.

    // Set up dictionary.
    NeoN::Database db;
    NeoN::Dictionary fvSchemes;
    NeoN::Dictionary ddtSchemes;
    ddtSchemes.insert("type", std::string("Runge-Kutta"));
    ddtSchemes.insert("Runge-Kutta-Method", std::string("Forward-Euler"));
    fvSchemes.insert("ddtSchemes", ddtSchemes);
    NeoN::Dictionary fvSolution;

    // Set up fields.
    auto mesh = NeoN::createSingleCellMesh(exec);
    fvcc::FieldCollection& fieldCollection = fvcc::FieldCollection::instance(db, "fieldCollection");
    fvcc::VolumeField<NeoN::scalar>& vf =
        fieldCollection.registerField<fvcc::VolumeField<NeoN::scalar>>(
            CreateField {.name = "vf", .mesh = mesh, .timeIndex = 1}
        );

    // Setup solve parameters.
    const NeoN::scalar maxTime = 0.1;
    const NeoN::scalar initialValue = 1.0;
    std::array<NeoN::scalar, 2> deltaTime = {0.01, 0.001};

    SECTION("Solve on " + execName)
    {
        int iTest = 0;
        std::array<NeoN::scalar, 2> error;
        for (auto dt : deltaTime)
        {
            // reset
            auto& vfOld = fvcc::oldTime(vf);
            NeoN::scalar time = 0.0;
            vf.internalField() = initialValue;
            vfOld.internalField() = initialValue;

            // Set expression
            TemporalOperator ddtOp = NeoN::dsl::imp::ddt(vfOld);

            auto divOp = YSquared(vfOld);
            auto eqn = ddtOp + divOp;

            // solve.
            while (time < maxTime)
            {
                NeoN::dsl::solve(eqn, vf, time, dt, fvSchemes, fvSolution);
                time += dt;
            }

            // check error.
            NeoN::scalar analytical = 1.0 / (initialValue - maxTime);
            auto vfHost = vf.internalField().copyToHost();
            error[iTest] = std::abs(vfHost.view()[0] - analytical);
            iTest++;
        }

        // check order of convergence.
        NeoN::scalar order = (std::log(error[0]) - std::log(error[1]))
                           / (std::log(deltaTime[0]) - std::log(deltaTime[1]));
        REQUIRE(order > (1.0 - convergenceTolerance));
    }
}
