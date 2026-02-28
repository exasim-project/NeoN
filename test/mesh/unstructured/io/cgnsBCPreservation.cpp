// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshWriter.hpp"

#include <filesystem>

TEST_CASE("Boundary patch name preservation")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("cube3D patch names survive round-trip " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);

        // Verify patch names were stored in stencilDB
        REQUIRE(mesh.stencilDB().contains("io::patchNames"));
        auto& names =
            mesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");
        REQUIRE(names->size() == 6);

        // Write and re-read
        NeoN::io::writeCgns(mesh, "output_bc_test.cgns");
        auto mesh2 = NeoN::io::readCgns("output_bc_test.cgns", exec);

        REQUIRE(mesh2.stencilDB().contains("io::patchNames"));
        auto& names2 =
            mesh2.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");
        REQUIRE(names2->size() == names->size());

        // Names must match (order preserved)
        for (std::size_t i = 0; i < names->size(); ++i)
        {
            REQUIRE((*names2)[i] == (*names)[i]);
        }

        std::filesystem::remove("output_bc_test.cgns");
    }

    SECTION("Face count per patch preserved " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        NeoN::io::writeCgns(mesh, "output_bc_faces.cgns");
        auto mesh2 = NeoN::io::readCgns("output_bc_faces.cgns", exec);

        auto const& off1 = mesh.boundaryMesh().offset();
        auto const& off2 = mesh2.boundaryMesh().offset();
        REQUIRE(off1.size() == off2.size());
        for (std::size_t i = 0; i < off1.size(); ++i)
        {
            REQUIRE(off1[i] == off2[i]);
        }

        std::filesystem::remove("output_bc_faces.cgns");
    }

    SECTION("singleTet patch names survive round-trip " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/singleTet.cgns", exec);

        REQUIRE(mesh.stencilDB().contains("io::patchNames"));
        auto& names =
            mesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");
        REQUIRE(names->size() == 4);

        NeoN::io::writeCgns(mesh, "output_tet_bc.cgns");
        auto mesh2 = NeoN::io::readCgns("output_tet_bc.cgns", exec);

        REQUIRE(mesh2.stencilDB().contains("io::patchNames"));
        auto& names2 =
            mesh2.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");
        REQUIRE(names2->size() == names->size());

        for (std::size_t i = 0; i < names->size(); ++i)
        {
            REQUIRE((*names2)[i] == (*names)[i]);
        }

        std::filesystem::remove("output_tet_bc.cgns");
    }
}
