// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

fvcc::VolumeField<NeoN::scalar>
createVolumeField(const NeoN::UnstructuredMesh& mesh, std::string fieldName)
{
    std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs {};
    for (auto patchi : std::vector<NeoN::localIdx> {0, 1, 2, 3})
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", 2.0);
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, dict, patchi));
    }
    fvcc::VolumeField<NeoN::scalar> vf(mesh.exec(), fieldName, mesh, bcs);
    NeoN::fill(vf.internalVector(), 1.0);
    return vf;
}

struct CreateVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;
    NeoN::Document operator()(NeoN::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs {};
        for (auto patchi : std::vector<NeoN::localIdx> {0, 1, 2, 3})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 2.0);
            bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, dict, patchi));
        }

        NeoN::Field<NeoN::scalar> domainVector(
            mesh.exec(),
            NeoN::Vector<NeoN::scalar>(mesh.exec(), mesh.nCells(), 1.0),
            {0, 10, 20, 30}
        );
        fvcc::VolumeField<NeoN::scalar> vf(mesh.exec(), name, mesh, domainVector, bcs, db, "", "");
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


TEST_CASE("Vector Document")
{
    NeoN::Executor exec = GENERATE(
        NeoN::Executor(NeoN::SerialExecutor {}),
        NeoN::Executor(NeoN::CPUExecutor {}),
        NeoN::Executor(NeoN::GPUExecutor {})
    );


    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);

    SECTION("create VectorDocument: " + execName)
    {

        fvcc::VectorDocument fieldDoc(createVolumeField(mesh, "T"), 1, 2, 3);


        SECTION("validate VectorDocument")
        {
            REQUIRE(fieldDoc.doc().validate());
            REQUIRE_NOTHROW(fieldDoc.doc().validate());

            REQUIRE(fieldDoc.doc().keys().size() == 6);
            REQUIRE(fieldDoc.id().substr(0, 4) == "doc_");
            REQUIRE(fieldDoc.timeIndex() == 1);
            REQUIRE(fieldDoc.iterationIndex() == 2);
            REQUIRE(fieldDoc.subCycleIndex() == 3);
            REQUIRE(fieldDoc.name() == "T");


            NeoN::Document& doc = fieldDoc.doc();
            REQUIRE(doc.validate());
            REQUIRE_NOTHROW(doc.validate());
            REQUIRE(doc.keys().size() == 6);
            REQUIRE(doc.id().substr(0, 4) == "doc_");


            // REQUIRE(name(doc) == "T");
            const auto& constVolVector = fieldDoc.field<fvcc::VolumeField<NeoN::scalar>>();
            auto& volVector = fieldDoc.field<fvcc::VolumeField<NeoN::scalar>>();

            REQUIRE(volVector.name == "T");
            auto volVectorHost = volVector.internalVector().copyToHost();
            REQUIRE(volVectorHost.view()[0] == 1.0);
            REQUIRE(&volVector == &constVolVector);
        }

        SECTION("modify fieldDocument")
        {
            fieldDoc.timeIndex() = 4;
            fieldDoc.iterationIndex() = 5;
            fieldDoc.subCycleIndex() = 6;
            auto& volVector = fieldDoc.field<fvcc::VolumeField<NeoN::scalar>>();
            NeoN::fill(volVector.internalVector(), 2.0);

            auto volVectorHost = volVector.internalVector().copyToHost();
            REQUIRE(volVectorHost.view()[0] == 2.0);
            REQUIRE(fieldDoc.timeIndex() == 4);
            REQUIRE(fieldDoc.iterationIndex() == 5);
            REQUIRE(fieldDoc.subCycleIndex() == 6);
        }
    }
}

TEST_CASE("VectorCollection")
{
    NeoN::Database db;

    NeoN::Executor exec = GENERATE(
        NeoN::Executor(NeoN::SerialExecutor {}),
        NeoN::Executor(NeoN::CPUExecutor {}),
        NeoN::Executor(NeoN::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);

    SECTION("create VectorCollection: " + execName)
    {
        fvcc::VectorCollection fieldCollection(db, "testVectorCollection");
        REQUIRE(fieldCollection.size() == 0);
    }

    SECTION("add VectorDocument to VectorCollection" + execName)
    {
        fvcc::VectorCollection& fieldCollection =
            fvcc::VectorCollection::instance(db, "testVectorCollection");
        REQUIRE(db.size() == 1);

        fvcc::VectorDocument fieldDoc(createVolumeField(mesh, "T1"), 1, 2, 3);

        REQUIRE(fieldCollection.insert(fieldDoc) != std::string(""));
        REQUIRE(
            fieldCollection.insert(fvcc::VectorDocument(createVolumeField(mesh, "T2"), 1, 2, 3))
            != std::string("")
        );
        REQUIRE(
            fieldCollection.insert(fvcc::VectorDocument(createVolumeField(mesh, "T3"), 1, 2, 3))
            != std::string("")
        );

        REQUIRE(fieldCollection.size() == 3);

        SECTION("get VectorDocument from VectorCollection")
        {
            fvcc::VectorDocument& doc = fieldCollection.fieldDoc(fieldDoc.id());
            REQUIRE(doc.doc().validate());
            REQUIRE(doc.doc().keys().size() == 6);
            REQUIRE(doc.id().substr(0, 4) == "doc_");
            REQUIRE(doc.timeIndex() == 1);
            REQUIRE(doc.iterationIndex() == 2);
            REQUIRE(doc.subCycleIndex() == 3);
            REQUIRE(doc.name() == "T1");

            const auto& constVolVector = doc.field<fvcc::VolumeField<NeoN::scalar>>();
            auto& volVector = doc.field<fvcc::VolumeField<NeoN::scalar>>();

            REQUIRE(volVector.name == "T1");
            auto volVectorHost = volVector.internalVector().copyToHost();
            REQUIRE(volVectorHost.view()[0] == 1.0);
            REQUIRE(&volVector == &constVolVector);
        }

        SECTION("query")
        {
            auto resTimeIndex =
                fieldCollection.find([](const NeoN::Document& doc)
                                     { return doc.get<std::int64_t>("timeIndex") == 1; });

            REQUIRE(resTimeIndex.size() == 3);

            auto resSubCycleIndex =
                fieldCollection.find([](const NeoN::Document& doc)
                                     { return doc.get<std::int64_t>("subCycleIndex") == 5; });

            REQUIRE(resSubCycleIndex.size() == 0);

            auto resName = fieldCollection.find([](const NeoN::Document& doc)
                                                { return doc.get<std::string>("name") == "T3"; });

            REQUIRE(resName.size() == 1);

            const auto& fieldDoc2 = fieldCollection.fieldDoc(resName[0]);
            REQUIRE(fieldDoc2.timeIndex() == 1);
        }
    }

    SECTION("register " + execName)
    {

        fvcc::VectorCollection& fieldCollection1 =
            fvcc::VectorCollection::instance(db, "newTestVectorCollection");
        REQUIRE(db.size() == 1);

        fvcc::VolumeField<NeoN::scalar>& t =
            fieldCollection1.registerVector<fvcc::VolumeField<NeoN::scalar>>(CreateVector {
                .name = "T", .mesh = mesh, .timeIndex = 1, .iterationIndex = 1, .subCycleIndex = 1
            });

        REQUIRE(t.name == "T");
        REQUIRE(t.hasDatabase());
        auto tHost = t.internalVector().copyToHost();
        REQUIRE(tHost.view()[0] == 1.0);
        REQUIRE(t.registered());

        SECTION("Construct from Vector")
        {
            fvcc::VectorCollection& fieldCollection2 = fvcc::VectorCollection::instance(t);
            REQUIRE(fieldCollection2.size() == 1);
            const fvcc::VolumeField<NeoN::scalar>& constT = t;
            const fvcc::VectorCollection& fieldCollection3 =
                fvcc::VectorCollection::instance(constT);
            REQUIRE(fieldCollection3.size() == 1);
        }


        SECTION("register from existing field")
        {
            fvcc::VectorCollection& fieldCollection2 = fvcc::VectorCollection::instance(t);
            fvcc::VolumeField<NeoN::scalar>& t3 =
                fieldCollection2.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                    fvcc::CreateFromExistingVector<fvcc::VolumeField<NeoN::scalar>> {
                        .name = "T3", .field = t
                    }
                );

            const fvcc::VectorDocument& docT = fieldCollection2.fieldDoc(t3.key);
            const fvcc::VectorDocument& docT3 = fieldCollection2.fieldDoc(t.key);

            REQUIRE(docT.timeIndex() == docT3.timeIndex());
            REQUIRE(docT.iterationIndex() == docT3.iterationIndex());
            REQUIRE(docT.subCycleIndex() == docT3.subCycleIndex());
        }
    }
}
