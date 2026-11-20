// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace dsl = NeoN::dsl;
using Catch::randomizeVector;

namespace NeoN
{

namespace
{

Dictionary schemes(const std::string& post)
{
    return Dictionary {
        {"laplacianSchemes",
         Dictionary {
             {"laplacian(gamma" + post + ",U" + post + ")",
              TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")})}
         }},
        {"divSchemes",
         Dictionary {
             {"div(phi" + post + ",U" + post + ")",
              TokenList({std::string("Gauss"), std::string("upwind")})}
         }}
    };
}

#if NF_WITH_GINKGO
// the solve has to stop before the iteration limit and reach the tolerance of the solver
// dictionaries below, otherwise matching results could also mean that both solves stopped at the
// iteration limit
void requireConverged(const la::SolverStats& stats)
{
    REQUIRE(stats.entries[0].numIter < 200);
    REQUIRE(stats.entries[0].finalResNorm <= 1e-10 * stats.entries[0].initResNorm);
}
#endif

}

// checks that the distributed ELL solve gives the same result as the distributed CSR solve and
// the serial solve. only GMRES without a preconditioner is tested here
TEST_CASE("Distributed ELL solve matches distributed CSR")
{
    using ELLMatrix = la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 12;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto u = fvcc::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );
    srand(42);
    randomizeVector(u);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = fvcc::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    auto gamma = fvcc::SurfaceField<scalar>(exec, "gamma", mesh, surfaceBCs);
    srand(42);
    randomizeVector(phi.internalVector());
    fill(gamma.internalVector(), 2.0);

    dsl::SetReference<scalar, localIdx> setRef(0, 0.0);

    auto expr = dsl::imp::div(phi, u) - dsl::imp::laplacian(gamma, u);
    expr.read(schemes(""));
    auto ls = expr.assemble(mesh, 1.0, 1.0, {&setRef});
    fill(ls.rhs(), 2.0);

    NeoN::mpi::Environment mpiEnviron;
    const localIdx nLocal = nCells / static_cast<localIdx>(mpiEnviron.sizeRank());
    auto meshPart = create1DUniformMeshPart(exec, nLocal);

    auto uPart = detail::oneDPartitionField(u, meshPart, mpiEnviron);
    auto phiPart = detail::oneDPartitionField(phi, meshPart, mpiEnviron);
    auto gammaPart = detail::oneDPartitionField(gamma, meshPart, mpiEnviron);

    auto exprDist = dsl::imp::div(phiPart, uPart) - dsl::imp::laplacian(gammaPart, uPart);
    exprDist.read(schemes("Part"));

    auto lsCsr = exprDist.assemble(meshPart, 1.0, 1.0, {&setRef});
    auto lsEll = exprDist.assemble<scalar, ELLMatrix>(meshPart, 1.0, 1.0, {&setRef});
    fill(lsCsr.rhs(), 2.0);
    fill(lsEll.rhs(), 2.0);

    SECTION("ELL carries the same processor coupling as CSR")
    {
        REQUIRE_FALSE(lsEll.commPattern().sendCounts.empty());
        REQUIRE_THAT(
            lsEll.offDiagonalMatrix().values(), Equals(lsCsr.offDiagonalMatrix().values())
        );
        REQUIRE_THAT(
            lsEll.offDiagonalMatrix().sparsity()->rowIdxs(),
            Equals(lsCsr.offDiagonalMatrix().sparsity()->rowIdxs(), EqualInt())
        );
        REQUIRE_THAT(
            lsEll.offDiagonalMatrix().sparsity()->colIdxs(),
            Equals(lsCsr.offDiagonalMatrix().sparsity()->colIdxs(), EqualInt())
        );
    }

#if NF_WITH_GINKGO
    SECTION("ELL solve matches CSR and serial")
    {
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Gmres"},
             {"criteria", Dictionary {{{"iteration", 200}, {"relative_residual_norm", 1e-10}}}}}
        };

        auto x = Vector<scalar>(exec, nCells, 0.0);
        auto xCsr = Vector<scalar>(exec, nLocal, 0.0);
        auto xEll = Vector<scalar>(exec, nLocal, 0.0);

        // use separate solvers so the ELL solve does not reuse the cached CSR matrix
        auto statsSerial = la::Solver(exec, solverDict).solve(ls, x);
        auto statsCsr = la::Solver(exec, solverDict).solve(lsCsr, xCsr);
        auto statsEll = la::Solver(exec, solverDict).solve(lsEll, xEll);

        requireConverged(statsSerial);
        requireConverged(statsCsr);
        requireConverged(statsEll);
        REQUIRE(statsEll.entries[0].numIter == statsCsr.entries[0].numIter);
        REQUIRE(
            statsEll.entries[0].finalResNorm
            == Catch::Approx(statsCsr.entries[0].finalResNorm).margin(1e-10)
        );

        REQUIRE_THAT(xEll, Equals(xCsr, Approx {1e-10}));

        const auto offset = nLocal * static_cast<localIdx>(mpiEnviron.rank());
        REQUIRE_THAT(xEll, Equals(take(x, {offset, offset + nLocal}), Approx {1e-6}));
    }
#endif
}

}
