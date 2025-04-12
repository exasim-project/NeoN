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
    for (auto patchi : std::vector<size_t> {0, 1, 2, 3})
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", 2.0);
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, dict, patchi));
    }
    fvcc::VolumeField<NeoN::scalar> vf(mesh.exec(), fieldName, mesh, bcs);
    NeoN::fill(vf.internalField(), 1.0);
    return vf;
}

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
        for (auto patchi : std::vector<size_t> {0, 1, 2, 3})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 2.0);
            bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, dict, patchi));
        }
        NeoN::DomainField<NeoN::scalar> domainField(
            mesh.exec(), NeoN::Field<NeoN::scalar>(mesh.exec(), mesh.nCells(), 1.0), {0, 10, 20, 30}
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

TEST_CASE("oldTimeCollection")
{
    NeoN::Database db;

    NeoN::Executor exec = GENERATE(
        NeoN::Executor(NeoN::SerialExecutor {}),
        NeoN::Executor(NeoN::CPUExecutor {}),
        NeoN::Executor(NeoN::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);

    SECTION("OldTimeDocument")
    {
        fvcc::OldTimeDocument oldTimeDoc("next", "previous", "current", 1);

        REQUIRE(oldTimeDoc.nextTime() == "next");
        REQUIRE(oldTimeDoc.previousTime() == "previous");
        REQUIRE(oldTimeDoc.currentTime() == "current");
        REQUIRE(oldTimeDoc.level() == 1);
        REQUIRE(oldTimeDoc.typeName() == "OldTimeDocument");
    }

    SECTION("OldTimeCollection from db and name")
    {
        fvcc::OldTimeCollection& oldTimeCollection = fvcc::OldTimeCollection::instance(
            db, "testFieldCollection_oldTime", "testFieldCollection"
        );

        REQUIRE(oldTimeCollection.name() == "testFieldCollection_oldTime");
        REQUIRE(oldTimeCollection.type() == "OldTimeDocument");
        REQUIRE(oldTimeCollection.db().contains("testFieldCollection_oldTime"));
        REQUIRE(oldTimeCollection.size() == 0);

        fvcc::OldTimeDocument oldTimeDoc("nextTime", "previousTime", "currentTime", 1);

        oldTimeCollection.insert(oldTimeDoc);
        std::string oldTimeDocKey = oldTimeDoc.id();

        REQUIRE(oldTimeCollection.size() == 1);
        REQUIRE(oldTimeCollection.contains(oldTimeDocKey));
        REQUIRE(oldTimeCollection.findNextTime("nextTime") == oldTimeDocKey);
        REQUIRE(oldTimeCollection.findPreviousTime("previousTime") == oldTimeDocKey);
    }

    SECTION("OldTimeCollection from FieldCollection")
    {
        fvcc::FieldCollection& fieldCollection =
            fvcc::FieldCollection::instance(db, "testFieldCollection");

        auto& oldTimeCollection = fvcc::OldTimeCollection::instance(fieldCollection);
        REQUIRE(oldTimeCollection.name() == "testFieldCollection_oldTime");
        REQUIRE(oldTimeCollection.type() == "OldTimeDocument");
        REQUIRE(oldTimeCollection.db().contains("testFieldCollection_oldTime"));
        REQUIRE(oldTimeCollection.size() == 0);

        const auto& oldTimeCollectionConst = fvcc::OldTimeCollection::instance(fieldCollection);
        REQUIRE(oldTimeCollectionConst.name() == "testFieldCollection_oldTime");
    }

    SECTION("oldTime")
    {
        fvcc::FieldCollection& fieldCollection =
            fvcc::FieldCollection::instance(db, "testFieldCollection");
        fvcc::VolumeField<NeoN::scalar>& t =
            fieldCollection.registerField<fvcc::VolumeField<NeoN::scalar>>(
                CreateField {.name = "T", .mesh = mesh, .timeIndex = 1}
            );
        // find the field by name and check the document key
        auto res = fieldCollection.find([](const NeoN::Document& doc) { return name(doc) == "T"; });

        REQUIRE(res.size() == 1);
        REQUIRE(t.key == res[0]);
        REQUIRE(t.fieldCollectionName == "testFieldCollection");

        REQUIRE(t.name == "T");
        REQUIRE(t.hasDatabase());
        auto tHost = t.internalField().copyToHost();
        REQUIRE(tHost.view()[0] == 1.0);
        REQUIRE(t.registered());
        fvcc::FieldDocument& docT = fieldCollection.fieldDoc(t.key);
        // without the timeIndex smaller than 2 the timeIndex overflows if we subtract 1
        // Note: we  are also checking oldTime oldTime(T) in the next section
        REQUIRE(docT.timeIndex() == 1);
        REQUIRE(docT.iterationIndex() == 0);
        REQUIRE(docT.subCycleIndex() == 0);


        SECTION("usage")
        {
            auto& tOld = fvcc::oldTime(t);

            REQUIRE(tOld.name == "T_0");
            auto tOldHost = tOld.internalField().copyToHost();
            REQUIRE(tOldHost.view()[0] == 1.0);
            fvcc::FieldDocument& doctOld = fieldCollection.fieldDoc(tOld.key);
            REQUIRE(doctOld.timeIndex() == 0);
            fvcc::OldTimeCollection& oldTimeCollection =
                fvcc::OldTimeCollection::instance(fieldCollection);
            REQUIRE(oldTimeCollection.size() == 1);
            auto& tOldDoc = oldTimeCollection.oldTimeDoc(oldTimeCollection.findNextTime(t.key));
            REQUIRE(tOldDoc.nextTime() == t.key);
            REQUIRE(tOldDoc.previousTime() == tOld.key);
            REQUIRE(tOldDoc.currentTime() == t.key);
            REQUIRE(tOldDoc.level() == 1);

            auto& sametOld = fvcc::oldTime(t);
            // check if the same field is returned
            REQUIRE(&tOld == &sametOld);

            auto& tOld2 = fvcc::oldTime(tOld);
            REQUIRE(tOld2.name == "T_0_0");
            auto tOld2Host = tOld2.internalField().copyToHost();
            REQUIRE(tOld2Host.view()[0] == 1.0);
            fvcc::FieldDocument& doctOld2 = fieldCollection.fieldDoc(tOld2.key);
            REQUIRE(doctOld2.timeIndex() == -1);
            REQUIRE(oldTimeCollection.size() == 2);
            auto& tOldDoc2 = oldTimeCollection.oldTimeDoc(oldTimeCollection.findNextTime(tOld.key));
            REQUIRE(tOldDoc2.nextTime() == tOld.key);
            REQUIRE(tOldDoc2.previousTime() == tOld2.key);
            REQUIRE(tOldDoc2.currentTime() == t.key);
            REQUIRE(tOldDoc2.level() == 2);

            auto& sametOld2 = fvcc::oldTime(tOld);
            // check if the same field is returned
            REQUIRE(&tOld2 == &sametOld2);
        }
    }
}
