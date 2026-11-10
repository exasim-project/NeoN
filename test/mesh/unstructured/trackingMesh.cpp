// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/pointInterpolation.hpp"
#include "NeoN/mesh/unstructured/trackingMesh.hpp"

using NeoN::label;
using NeoN::localIdx;
using NeoN::scalar;
using NeoN::Vec3;

TEST_CASE("pointToBarycentric")
{
    // unit tet a = origin, b, c, d on the axes: det = 1
    const Vec3 a(0, 0, 0), b(1, 0, 0), c(0, 1, 0), d(0, 0, 1);
    scalar w[4];

    SECTION("a vertex has weight 1")
    {
        REQUIRE(NeoN::pointToBarycentric(a, a, b, c, d, w) == Catch::Approx(1.0));
        REQUIRE(w[0] == Catch::Approx(1.0));
        REQUIRE(w[1] == Catch::Approx(0.0).margin(1e-15));

        NeoN::pointToBarycentric(b, a, b, c, d, w);
        REQUIRE(w[0] == Catch::Approx(0.0).margin(1e-15));
        REQUIRE(w[1] == Catch::Approx(1.0));
    }

    SECTION("the centroid has equal weights 1/4")
    {
        NeoN::pointToBarycentric(Vec3(0.25, 0.25, 0.25), a, b, c, d, w);
        for (int j = 0; j < 4; j++)
        {
            REQUIRE(w[j] == Catch::Approx(0.25));
        }
    }

    SECTION("a flat (degenerate) tet returns det 0 and w = (1, 0, 0, 0)")
    {
        const Vec3 dFlat(1, 1, 0); // in the plane of a, b, c
        REQUIRE(NeoN::pointToBarycentric(Vec3(0.3, 0.3, 0), a, b, c, dFlat, w) == 0.0);
        REQUIRE(w[0] == 1.0);
        REQUIRE(w[1] == 0.0);
        REQUIRE(w[2] == 0.0);
        REQUIRE(w[3] == 0.0);
    }
}

TEST_CASE("TrackingMesh pointInCell and locateCell")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // unit square cell (0,0)-(1,1) with 4 boundary faces, see createSingleCellMesh
    NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);

    // connectivity of the single cell, written by hand: faces 0..3 of cell 0,
    // face f between points f and f+1, every face a domain boundary (escape)
    const label escape = label(NeoN::TrackingFaceType::escape);
    NeoN::TrackingMesh trackingMesh(
        mesh,
        NeoN::SegmentedVector<label, localIdx>(
            NeoN::Vector<label>(exec, {0, 1, 2, 3}), NeoN::Vector<localIdx>(exec, {0, 4})
        ),
        NeoN::SegmentedVector<label, localIdx>(
            NeoN::Vector<label>(exec, {0, 1, 1, 2, 2, 3, 3, 0}),
            NeoN::Vector<localIdx>(exec, {0, 2, 4, 6, 8})
        ),
        NeoN::Vector<label>(exec, 4, label(0)),
        NeoN::Vector<label>(exec, 4, escape),
        NeoN::Vector<label>(exec, 4, label(-1)),
        NeoN::Vector<Vec3>(exec, 4, Vec3(0, 0, 0)),
        NeoN::Vector<Vec3>(exec, {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}})
    );

    // the functions read executor memory, so they are called inside a kernel
    NeoN::Vector<label> result(exec, 4, label(-99));
    auto resultV = result.view();
    const NeoN::TrackingMeshView m = trackingMesh.view();
    NeoN::parallelFor(
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx) {
            Vec3 inside(0.25, 0.75, 0.0);
            Vec3 outside(2.0, 0.5, 0.0);
            resultV[0] = NeoN::pointInCell(inside, 0, m) ? 1 : 0;
            resultV[1] = NeoN::pointInCell(outside, 0, m) ? 1 : 0;
            resultV[2] = NeoN::locateCell(inside, 0, m);
            resultV[3] = NeoN::locateCell(outside, 0, m);
        },
        "testTrackingMesh"
    );
    const auto resultH = result.copyToHost();

    SECTION("pointInCell: inside and outside the cell (" + execName + ")")
    {
        REQUIRE(resultH.view()[0] == 1);
        REQUIRE(resultH.view()[1] == 0);
    }

    SECTION("locateCell: inside returns the cell, outside returns -1 (" + execName + ")")
    {
        REQUIRE(resultH.view()[2] == 0);
        REQUIRE(resultH.view()[3] == -1);
    }
}
