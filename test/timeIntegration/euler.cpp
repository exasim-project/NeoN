// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "../dsl/common.hpp"

#include "NeoN/NeoN.hpp"

// only needed for msvc
template class NeoN::timeIntegration::ForwardEuler<VolumeField>;

struct CreateVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    NeoN::scalar value = 0;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs {};
        NeoN::Field<NeoN::scalar> domainVector(
            mesh.exec(),
            NeoN::Vector<NeoN::scalar>(mesh.exec(), mesh.nCells(), 1.0),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<NeoN::scalar> vf(mesh.exec(), name, mesh, domainVector, bcs, db, "", "");
        NeoN::fill(vf.internalVector(), value);
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

TEST_CASE("TimeIntegration")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Database db;
    auto mesh = NeoN::createSingleCellMesh(exec);
    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "fieldCollection");


    fvcc::VolumeField<NeoN::scalar>& vf =
        fieldCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
            CreateVector {.name = "vf", .mesh = mesh, .value = 2.0, .timeIndex = 1}
        );

    auto dummy = Dummy(vf);

    double dt {2.0};
    double time {1.0};

    NeoN::Dictionary fvSolution;
    NeoN::Dictionary fvSchemes;
    NeoN::Dictionary ddtSchemes;

    SECTION("ForwardEuler via DSL on " + execName)
    {
        ddtSchemes.insert("type", std::string("forwardEuler"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        // ddt(U) = f
        NeoN::dsl::Expression<NeoN::scalar> eqn = NeoN::dsl::exp::ddt(vf) + dummy;

        // int(ddt(U)) + f = 0
        // (U^1-U^0)/dt = -f
        // U^1 = - f * dt + U^0, where dt = 2, f = 2, U^0=2.0 -> U^1=-2.0
        NeoN::dsl::solve(eqn, vf, time, dt, fvSchemes, fvSolution);
        REQUIRE(getVector(vf.internalVector()) == -2.0);
    }

    SECTION("BackwardEuler via DSL on " + execName)
    {
        ddtSchemes.insert("type", std::string("backwardEuler"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        // ddt(U) = f
        NeoN::dsl::Expression<NeoN::scalar> eqn = NeoN::dsl::imp::ddt(vf) + dummy;

        // int(ddt(U)) + f = 0
        // (U^1-U^0)/dt = -f
        // U^1 = - f * dt + U^0, where dt = 2, f = 2, U^0=2.0 -> U^1=-2.0
        NeoN::dsl::solve(eqn, vf, time, dt, fvSchemes, fvSolution);
        REQUIRE(getVector(vf.internalVector()) == -5.0);
    }
}
