// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <string>

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Vector;

// These tests prove that createEmptyLinearSystem caches and SHARES a single immutable,
// topology-only sparsity bundle (CSR system sparsity + FaceToMatrixAddress + COO boundary
// sparsity) per mesh, while keeping the per-system value vectors independent. This is a pure
// memory/aliasing refactor — there must be zero numerical change.
TEST_CASE("SharedSparsity")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const auto nCells = 10;

    SECTION(
        "shares system/address/boundary sparsity pointers across two systems on same mesh "
        + execName
    )
    {
        auto mesh = create1DUniformMesh(exec, nCells);

        auto lsA = NeoN::la::createEmptyLinearSystem<scalar>(mesh);
        auto lsB = NeoN::la::createEmptyLinearSystem<scalar>(mesh);

        // identical system sparsity pointer
        REQUIRE(lsA.matrix().sparsity().get() == lsB.matrix().sparsity().get());
        // identical FaceToMatrixAddress pointer
        REQUIRE(
            lsA.matrix().faceToMatrixAddress().get() == lsB.matrix().faceToMatrixAddress().get()
        );
        // identical boundary sparsity pointer
        REQUIRE(lsA.boundaryMatrix().sparsity().get() == lsB.boundaryMatrix().sparsity().get());
    }

    SECTION("keeps values independent across two systems on same mesh " + execName)
    {
        auto mesh = create1DUniformMesh(exec, nCells);

        auto lsA = NeoN::la::createEmptyLinearSystem<scalar>(mesh);
        auto lsB = NeoN::la::createEmptyLinearSystem<scalar>(mesh);

        // The value vectors must be distinct allocations.
        REQUIRE(lsA.matrix().values().data() != lsB.matrix().values().data());

        // Mutate every value of lsA and confirm lsB is unchanged (no aliasing of values_).
        auto aView = lsA.matrix().values().view();
        parallelFor(
            exec, {0, aView.size()}, NEON_LAMBDA(const localIdx i) { aView[i] = scalar {42.0}; }
        );

        auto bHost = lsB.matrix().values().copyToHost();
        auto bHostV = bHost.view();
        for (localIdx i = 0; i < bHostV.size(); ++i)
        {
            REQUIRE(bHostV[i] == scalar {0.0});
        }
    }

    SECTION("distinct meshes get distinct sparsity pointers " + execName)
    {
        auto meshA = create1DUniformMesh(exec, nCells);
        auto meshB = create1DUniformMesh(exec, nCells);

        auto lsA = NeoN::la::createEmptyLinearSystem<scalar>(meshA);
        auto lsB = NeoN::la::createEmptyLinearSystem<scalar>(meshB);

        // The two meshes own independent stencilDB caches → distinct cached pointers.
        REQUIRE(lsA.matrix().sparsity().get() != lsB.matrix().sparsity().get());
        REQUIRE(
            lsA.matrix().faceToMatrixAddress().get() != lsB.matrix().faceToMatrixAddress().get()
        );
        REQUIRE(lsA.boundaryMatrix().sparsity().get() != lsB.boundaryMatrix().sparsity().get());
    }

    SECTION("scalar and Vec3 systems share the topology bundle on the same mesh " + execName)
    {
        auto mesh = create1DUniformMesh(exec, nCells);

        auto lsScalar = NeoN::la::createEmptyLinearSystem<scalar>(mesh);
        auto lsVec3 = NeoN::la::createEmptyLinearSystem<NeoN::Vec3>(mesh);

        // Topology is value-type-independent: the bundle is keyed on the sparsity type, which is
        // CsrSparsityPattern<localIdx> for both scalar and Vec3 systems → same cached pointers.
        REQUIRE(lsScalar.matrix().sparsity().get() == lsVec3.matrix().sparsity().get());
        REQUIRE(
            lsScalar.matrix().faceToMatrixAddress().get()
            == lsVec3.matrix().faceToMatrixAddress().get()
        );
        REQUIRE(
            lsScalar.boundaryMatrix().sparsity().get() == lsVec3.boundaryMatrix().sparsity().get()
        );
    }
}
