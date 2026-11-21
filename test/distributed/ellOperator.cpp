// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

#if NF_WITH_GINKGO
#include <filesystem>
#include <fstream>

#include "NeoN/linearAlgebra/ginkgo.hpp"
#endif

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

// every rank writes its own file, so the ranks do not write to the same file at once
std::string writeConfig(const std::string& name, const std::string& json, int rank)
{
    auto path = std::filesystem::temp_directory_path()
              / ("neon_ellOperator_" + name + "_" + std::to_string(rank) + ".json");
    std::ofstream out(path);
    out << json;
    return path.string();
}

// the exact ILU factorization, the default ParIlu is not deterministic on GPUs, so the CSR and the
// ELL solve could take a different number of iterations
const std::string schwarzIluJson = R"({"type": "solver::Cg",
    "criteria": [{"type": "Iteration", "max_iters": 200},
                 {"type": "ResidualNorm", "baseline": "rhs_norm", "reduction_factor": 1e-11}],
    "preconditioner": {"type": "preconditioner::Schwarz",
                       "local_solver": {"type": "preconditioner::Ilu",
                                        "factorization": {"type": "factorization::Ilu"}}}})";

// one scale corrected MergedPgm V-cycle as preconditioner of CG, the scale correction takes the
// diagonal of the local block of the distributed matrix
const std::string mergedPgmJson = R"({"type": "solver::Cg",
    "criteria": [{"type": "Iteration", "max_iters": 200},
                 {"type": "ResidualNorm", "baseline": "rhs_norm", "reduction_factor": 1e-11}],
    "preconditioner": {"type": "solver::Multigrid",
        "mg_level": [{"type": "neon::MergedPgm", "merge_levels": 1, "deterministic": true,
                      "scale_correction": true}],
        "pre_smoother": [{"type": "solver::Ir",
            "solver": {"type": "preconditioner::Schwarz",
                       "local_solver": {"type": "preconditioner::Jacobi", "max_block_size": 1}},
            "relaxation_factor": 0.6666666666666666,
            "criteria": [{"type": "Iteration", "max_iters": 2}]}],
        "post_smoother": [{"type": "solver::Ir",
            "solver": {"type": "preconditioner::Schwarz",
                       "local_solver": {"type": "preconditioner::Jacobi", "max_block_size": 1}},
            "relaxation_factor": 0.6666666666666666,
            "criteria": [{"type": "Iteration", "max_iters": 2}]}],
        "coarsest_solver": [{"type": "solver::Cg",
            "criteria": [{"type": "Iteration", "max_iters": 50},
                         {"type": "ResidualNorm", "reduction_factor": 1e-4}]}],
        "min_coarse_rows": 16,
        "cycle": "v",
        "criteria": [{"type": "Iteration", "max_iters": 1}]}})";
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

    SECTION("ELL solve after reassembly matches a fresh solve")
    {
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Gmres"},
             {"criteria", Dictionary {{{"iteration", 200}, {"relative_residual_norm", 1e-10}}}}}
        };

        // first solve fills the cached distributed matrix of this solver
        auto solver = la::Solver(exec, solverDict);
        auto xFirst = Vector<scalar>(exec, nLocal, 0.0);
        requireConverged(solver.solve(lsEll, xFirst));

        const auto* valuesPtr = lsEll.matrix().values().data();
        const auto offDiagBefore = lsEll.offDiagonalMatrix().values().copyToHost();

        // change the coefficients and reassemble into the same linear system, so the second solve
        // reuses the cached matrix and only has to refresh the off-diagonal values
        gammaPart.internalVector() *= 2.5;
        gammaPart.boundaryData().value() *= 2.5;
        lsEll.reset();
        exprDist.assemble<scalar, ELLMatrix>(1.0, 1.0, lsEll, meshPart, {&setRef});
        fill(lsEll.rhs(), 2.0);

        REQUIRE(lsEll.matrix().values().data() == valuesPtr);
        if (lsEll.offDiagonalMatrix().values().size() > 0)
        {
            REQUIRE_THAT(
                lsEll.offDiagonalMatrix().values(), !Equals(offDiagBefore, Approx {1e-12})
            );
        }

        auto xReused = Vector<scalar>(exec, nLocal, 0.0);
        auto statsReused = solver.solve(lsEll, xReused);

        // same system assembled and solved from scratch to compare against
        auto lsFresh = exprDist.assemble<scalar, ELLMatrix>(meshPart, 1.0, 1.0, {&setRef});
        fill(lsFresh.rhs(), 2.0);
        auto xFresh = Vector<scalar>(exec, nLocal, 0.0);
        auto statsFresh = la::Solver(exec, solverDict).solve(lsFresh, xFresh);

        requireConverged(statsReused);
        requireConverged(statsFresh);
        REQUIRE(statsReused.entries[0].numIter == statsFresh.entries[0].numIter);
        REQUIRE_THAT(xReused, Equals(xFresh, Approx {1e-10}));
        REQUIRE_THAT(xReused, !Equals(xFirst, Approx {1e-6}));
    }

    SECTION("createGkoMtxDist reuses the distributed matrix for unchanged ELL storage")
    {
        using DistMtx = gko::experimental::distributed::Matrix<scalar, label, gko::int64>;
        using GkoEll = gko::matrix::Ell<scalar, localIdx>;

        const auto& commPattern = lsEll.commPattern();
        auto comm = gko::experimental::mpi::communicator(
            commPattern.env.comm(), !commPattern.env.gpuAwareMpi()
        );
        auto gkoExec = la::ginkgo::getGkoExecutor(exec);

        // the caches a GinkgoSolver keeps between solves
        std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>> imap;
        std::shared_ptr<gko::matrix::Coo<scalar, localIdx>> nonLocal;
        std::shared_ptr<const gko::LinOp> distMtx;
        const scalar* localValPtr = nullptr;

        auto create = [&](const la::ELLMatrix<scalar, localIdx>& mtx)
        {
            return la::ginkgo::createGkoMtxDist(
                gkoExec,
                comm,
                mtx,
                lsEll.offDiagonalMatrix(),
                commPattern,
                imap,
                nonLocal,
                distMtx,
                localValPtr
            );
        };

        auto first = create(lsEll.matrix());
        const auto imapFirst = imap;

        // the local block is a view of the NeoN storage, nothing is copied
        auto local = gko::as<const GkoEll>(gko::as<const DistMtx>(first)->get_local_matrix());
        REQUIRE(local->get_const_values() == lsEll.matrix().values().data());
        REQUIRE(local->get_const_col_idxs() == lsEll.matrix().sparsity()->colIdxs().data());

        // unchanged storage gives the same distributed matrix, only the off-diagonal values are
        // copied into it
        lsEll.offDiagonalMatrix().values() *= 2.0;
        auto second = create(lsEll.matrix());
        REQUIRE(second.get() == first.get());

        const auto nNonLocal = static_cast<gko::size_type>(nonLocal->get_num_stored_elements());
        auto nonLocalHost = gko::array<scalar>(
            gkoExec->get_master(),
            gko::array<scalar>::const_view(gkoExec, nNonLocal, nonLocal->get_const_values())
                .copy_to_array()
        );
        const std::vector<scalar> nonLocalValues(
            nonLocalHost.get_const_data(), nonLocalHost.get_const_data() + nNonLocal
        );
        REQUIRE_THAT(lsEll.offDiagonalMatrix().values(), Equals(nonLocalValues));

        // a new value buffer gives a new distributed matrix around the new storage, the index
        // map is kept
        auto mtxCopy =
            la::ELLMatrix<scalar, localIdx>(lsEll.matrix().values(), lsEll.matrix().sparsity());
        REQUIRE(mtxCopy.values().data() != lsEll.matrix().values().data());

        auto third = create(mtxCopy);
        REQUIRE(third.get() != first.get());
        REQUIRE(imap.get() == imapFirst.get());
        auto localCopy = gko::as<const GkoEll>(gko::as<const DistMtx>(third)->get_local_matrix());
        REQUIRE(localCopy->get_const_values() == mtxCopy.values().data());
    }
#endif
}

#if NF_WITH_GINKGO
// the preconditioners take the local block of the distributed matrix, which is an Ell instead of a
// Csr for the ELL system. checks that they give the same solve as for the CSR system
TEST_CASE("Distributed ELL solve with preconditioners matches distributed CSR")
{
    using ELLMatrix = la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::mpi::Environment mpiEnviron;

    // large enough for a few multigrid levels on each rank
    const localIdx nCells = 300;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto u = fvcc::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );

    // zero flux, so only the symmetric laplacian is left for CG
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = fvcc::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    auto gamma = fvcc::SurfaceField<scalar>(exec, "gamma", mesh, surfaceBCs);
    fill(phi.internalVector(), 0.0);
    fill(gamma.internalVector(), 2.0);

    dsl::SetReference<scalar, localIdx> setRef(0, 0.0);

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

    auto [name, json] = GENERATE(
        std::pair {std::string("schwarzIlu"), schwarzIluJson},
        std::pair {std::string("mergedPgm"), mergedPgmJson}
    );

    SECTION(name)
    {
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"configFile", writeConfig(name, json, static_cast<int>(mpiEnviron.rank()))}}
        };

        auto xCsr = Vector<scalar>(exec, nLocal, 0.0);
        auto xEll = Vector<scalar>(exec, nLocal, 0.0);

        auto statsCsr = la::Solver(exec, solverDict).solve(lsCsr, xCsr);
        auto statsEll = la::Solver(exec, solverDict).solve(lsEll, xEll);

        requireConverged(statsCsr);
        requireConverged(statsEll);
        REQUIRE(statsEll.entries[0].numIter == statsCsr.entries[0].numIter);
        REQUIRE_THAT(xEll, Equals(xCsr, Approx {1e-10}));
    }
}
#endif

}
