// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace dsl = NeoN::dsl;
using Catch::randomizeVector;

namespace NeoN
{

auto generateInput = [](std::string scheme, std::string post)
{
    auto constructDiv = [](auto post) { return "div(phi" + post + ",U" + post + ")"; };
    auto constructGamma = [](auto post) { return "laplacian(gamma" + post + ",U" + post + ")"; };

    return NeoN::Dictionary {
        {
            "laplacianSchemes",
            NeoN::Dictionary {
                {constructGamma(post),
                 NeoN::TokenList(
                     {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                 )}
            },
        },
        {"divSchemes",
         NeoN::Dictionary {{constructDiv(post), NeoN::TokenList({std::string("Gauss"), scheme})}}}
    };
};

TEST_CASE("Distributed Operator")
{
    float epsilon = 1e-32;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto input = generateInput("upwind", "");
    auto inputPart = generateInput("upwind", "Part");

    auto nCells = 12;
    auto meshGlobal = create1DUniformMesh(exec, nCells);
    auto mesh = create1DUniformMesh(exec, nCells);

    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto U = finiteVolume::cellCentred::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );
    auto p = finiteVolume::cellCentred::VolumeField<scalar>(
        exec, "p", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );

    srand(42);
    randomizeVector(U);
    randomizeVector(p);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    auto gamma = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "gamma", mesh, surfaceBCs);

    fill(phi.internalVector(), 1.0);
    srand(42);
    randomizeVector(phi.internalVector());
    fill(gamma.internalVector(), 2.0);

    // assembly on global mesh
    auto expr = dsl::imp::div(phi, U) - dsl::imp::laplacian(gamma, U);
    expr.read(input);
    dsl::SetReference<scalar, localIdx> setRef(0, 0.0);
    auto ls = expr.assemble(mesh, 1.0, 1.0, {&setRef});

    NeoN::mpi::Environment mpiEnviron;
    auto meshPart = create1DUniformMeshPart(exec, meshGlobal.nCells() / mpiEnviron.sizeRank());

    auto uPart = detail::oneDPartitionField(U, meshPart, mpiEnviron);
    auto pPart = detail::oneDPartitionField(p, meshPart, mpiEnviron);
    auto phiPart = detail::oneDPartitionField(phi, meshPart, mpiEnviron);
    auto gammaPart = detail::oneDPartitionField(gamma, meshPart, mpiEnviron);

    auto exprDist = dsl::imp::div(phiPart, uPart) - dsl::imp::laplacian(gammaPart, uPart);

    exprDist.read(inputPart);

    auto lsDst = exprDist.assemble(meshPart, 1.0, 1.0, {&setRef});

    fill(ls.rhs(), 2.0);
    fill(lsDst.rhs(), 2.0);

    localIdx firstElement = 0;
    localIdx lastElement = 0;
    SECTION_IF(mpiEnviron.rank() == 0, "Correct mtx on rank 0")
    {
        lastElement = 10;
        REQUIRE_THAT(
            lsDst.matrix().values(), Equals(take(ls.matrix().values(), {firstElement, lastElement}))
        );
    }
    SECTION_IF(mpiEnviron.rank() == 1, "Correct mtx on rank 1")
    {
        firstElement = 12;
        lastElement = 22;
        REQUIRE_THAT(
            lsDst.matrix().values(), Equals(take(ls.matrix().values(), {firstElement, lastElement}))
        );
    }
    SECTION_IF(mpiEnviron.rank() == 2, "Correct mtx on rank 2")
    {
        firstElement = 24;
        lastElement = 34;
        REQUIRE_THAT(
            lsDst.matrix().values(), Equals(take(ls.matrix().values(), {firstElement, lastElement}))
        );
    }

    // The global 12-cell 1D CSR matrix has 34 values (row 0: 2 entries, rows 1-10: 3 each,
    // row 11: 2 entries). Proc-boundary off-diagonal entries sit at:
    //   off(3,4)=global[10], off(4,3)=global[11], off(7,8)=global[22], off(8,7)=global[23].
    // These are excluded from each rank's main matrix and stored in offDiagonalMatrix instead.
    SECTION("Correct offDiagonalMatrix on partitioned mesh")
    {
        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 offDiagonalMatrix matches global off(3,4)")
        {
            REQUIRE_THAT(
                lsDst.offDiagonalMatrix().values(), Equals(take(ls.matrix().values(), {10, 11}))
            );
            std::shared_ptr<const NeoN::la::CooSparsityPattern<localIdx>> cooSparsity =
                lsDst.offDiagonalMatrix().sparsity();
            REQUIRE_THAT(cooSparsity->rowIdxs(), Equals(I {3}, EqualInt()));
            REQUIRE_THAT(cooSparsity->colIdxs(), Equals(I {4}, EqualInt()));
        }
        SECTION_IF(
            mpiEnviron.rank() == 1, "Rank 1 offDiagonalMatrix matches global off(4,3) and off(7,8)"
        )
        {
            auto globalHost = ls.matrix().values().copyToHost();
            auto globalView = globalHost.view();
            std::vector<scalar> expected = {globalView[11], globalView[22]};
            REQUIRE_THAT(lsDst.offDiagonalMatrix().values(), Equals(expected));
            std::shared_ptr<const NeoN::la::CooSparsityPattern<localIdx>> cooSparsity =
                lsDst.offDiagonalMatrix().sparsity();
            REQUIRE_THAT(cooSparsity->rowIdxs(), Equals(I {4, 7}, EqualInt()));
            REQUIRE_THAT(cooSparsity->colIdxs(), Equals(I {3, 8}, EqualInt()));
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 2 offDiagonalMatrix matches global off(8,7)")
        {
            REQUIRE_THAT(
                lsDst.offDiagonalMatrix().values(), Equals(take(ls.matrix().values(), {23, 24}))
            );
            std::shared_ptr<const NeoN::la::CooSparsityPattern<localIdx>> cooSparsity =
                lsDst.offDiagonalMatrix().sparsity();
            REQUIRE_THAT(cooSparsity->rowIdxs(), Equals(I {8}, EqualInt()));
            REQUIRE_THAT(cooSparsity->colIdxs(), Equals(I {7}, EqualInt()));
        }
    }

#if NF_WITH_GINKGO
    Dictionary solverDict {
        {{"solver", std::string {"Ginkgo"}},
         {"type", "solver::Gmres"},
         {"criteria", Dictionary {{{"iteration", 200}, {"relative_residual_norm", 1e-7}}}}}
    };

    auto solver = NeoN::la::Solver(exec, solverDict);
    auto x = Vector<scalar>(exec, 12);
    fill(x, 0.0);

    auto xPart = Vector<scalar>(exec, 4);
    fill(xPart, 0.0);

    auto solverStats = solver.solve(ls, x);
    auto solverStatsDist = solver.solve(lsDst, xPart);

    scalar finalResNorm = solverStats.entries[0].finalResNorm;
    scalar finalResNormDist = solverStatsDist.entries[0].finalResNorm;

    REQUIRE(finalResNormDist == Catch::Approx(finalResNorm).margin(1e-6));

    SECTION_IF(mpiEnviron.rank() == 0, "Correct solution on rank 0")
    {
        REQUIRE_THAT(xPart, Equals(take(x, {0, 4}), Approx {1e-4}));
    }
    SECTION_IF(mpiEnviron.rank() == 1, "Correct solution on rank 1")
    {
        REQUIRE_THAT(xPart, Equals(take(x, {4, 8}), Approx {1e-4}));
    }
    SECTION_IF(mpiEnviron.rank() == 2, "Correct solution on rank 2")
    {
        REQUIRE_THAT(xPart, Equals(take(x, {8, 12}), Approx {1e-4}));
    }
#endif
}

}
