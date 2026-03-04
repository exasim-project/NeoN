// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#ifdef NF_WITH_MPI_SUPPORT
#if NF_WITH_GINKGO

#include <chrono>
#include <vector>

#include <mpi.h>
#include <ginkgo/ginkgo.hpp>

#include "NeoN/linearAlgebra/distributedGinkgoSolver.hpp"
#include "NeoN/linearAlgebra/ginkgo.hpp"

namespace NeoN::la
{

DistributedGinkgoSolver::DistributedGinkgoSolver(
    const Executor& exec, const Dictionary& solverCfg, const UnstructuredMesh& mesh
)
    : gkoExec_(ginkgo::getGkoExecutor(exec)), mesh_(mesh), cfg_(solverCfg)
{}

SolverStats DistributedGinkgoSolver::solve(const LinearSystem<scalar>& ls, Vector<scalar>& x) const
{
    // -----------------------------------------------------------------------
    // 1. Extract partition metadata
    // -----------------------------------------------------------------------
    const auto& db = mesh_.stencilDB();
    const localIdx nLocalCells = mesh_.nCells();
    const label nGlobalCells =
        static_cast<label>(*db.get<std::shared_ptr<localIdx>>("partition::nGlobalCells"));

    const auto& globalCellIds =
        *db.get<std::shared_ptr<std::vector<localIdx>>>("partition::globalCellIds");
    const auto& ghostCellGlobalIds =
        *db.get<std::shared_ptr<std::vector<localIdx>>>("partition::ghostCellGlobalIds");

    // Convert localIdx vectors to label (Ginkgo uses label)
    std::vector<label> globalCellIdsL(globalCellIds.begin(), globalCellIds.end());
    std::vector<label> ghostCellGlobalIdsL(ghostCellGlobalIds.begin(), ghostCellGlobalIds.end());

    // -----------------------------------------------------------------------
    // 2. Convert local CSR (with ghost columns) → Ginkgo matrix_data
    // -----------------------------------------------------------------------
    auto matData = toGlobalMatrixData(ls.matrix(), globalCellIdsL, ghostCellGlobalIdsL, nLocalCells);
    matData.size = gko::dim<2> {
        static_cast<std::size_t>(nGlobalCells), static_cast<std::size_t>(nGlobalCells)
    };

    // Build RHS global matrix_data
    gko::matrix_data<scalar, label> rhsData {
        gko::dim<2> {static_cast<std::size_t>(nGlobalCells), 1}
    };
    {
        auto rhsHost = ls.rhs().copyToHost();
        const scalar* rhsPtr = rhsHost.data();
        for (localIdx i = 0; i < nLocalCells; ++i)
        {
            rhsData.nonzeros.emplace_back(globalCellIdsL[static_cast<std::size_t>(i)], 0, rhsPtr[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. Build Ginkgo distributed::Partition from local cell counts
    // -----------------------------------------------------------------------
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const int nRanks = comm.size();

    // Gather local cell counts using 64-bit integers to avoid MPI type mismatch
    std::int64_t localCount64 = static_cast<std::int64_t>(nLocalCells);
    std::vector<std::int64_t> allCounts64(static_cast<std::size_t>(nRanks));
    MPI_Allgather(
        &localCount64, 1, MPI_INT64_T, allCounts64.data(), 1, MPI_INT64_T, MPI_COMM_WORLD
    );

    // Build partition range bounds [0, count0, count0+count1, ...]
    std::vector<label> rangeBounds(static_cast<std::size_t>(nRanks + 1));
    rangeBounds[0] = 0;
    for (int r = 0; r < nRanks; ++r)
    {
        rangeBounds[static_cast<std::size_t>(r + 1)] = rangeBounds[static_cast<std::size_t>(r)]
                                                        + static_cast<label>(allCounts64[static_cast<std::size_t>(r)]);
    }

    using part_type = gko::experimental::distributed::Partition<label, label>;
    auto partition = gko::share(
        part_type::build_from_contiguous(gkoExec_, gko::array<label>::view(gkoExec_, nRanks + 1, rangeBounds.data()))
    );

    // -----------------------------------------------------------------------
    // 4. Create distributed matrix and vectors
    // -----------------------------------------------------------------------
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, label>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    auto A = gko::share(dist_mtx::create(gkoExec_, comm));
    A->read_distributed(matData, partition);

    auto b = dist_vec::create(gkoExec_, comm);
    b->read_distributed(rhsData, partition);

    gko::matrix_data<scalar, label> zeroData {
        gko::dim<2> {static_cast<std::size_t>(nGlobalCells), 1}
    };
    auto xDist = dist_vec::create(gkoExec_, comm);
    xDist->read_distributed(zeroData, partition);

    // -----------------------------------------------------------------------
    // 5. Solve with CG + Schwarz(Jacobi)
    // -----------------------------------------------------------------------
    using cg = gko::solver::Cg<scalar>;
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<scalar, label, label>;
    using jacobi = gko::preconditioner::Jacobi<scalar, label>;

    auto t0 = std::chrono::steady_clock::now();

    auto residualLogger = gko::share(gko::log::Convergence<scalar>::create());

    auto solver =
        cg::build()
            .with_preconditioner(schwarz::build().with_local_solver(jacobi::build()))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(500u),
                gko::stop::ResidualNorm<scalar>::build().with_reduction_factor(scalar(1e-10))
            )
            .on(gkoExec_)
            ->generate(A);
    solver->add_logger(residualLogger);
    solver->apply(b, xDist);

    auto t1 = std::chrono::steady_clock::now();
    scalar solveTime = static_cast<scalar>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
    ) * scalar(1e-6);

    // -----------------------------------------------------------------------
    // 6. Copy local solution back into x
    // -----------------------------------------------------------------------
    auto localVec = xDist->get_local_vector();
    scalar* xPtr = x.data();
    for (localIdx i = 0; i < nLocalCells; ++i)
    {
        xPtr[i] = localVec->at(i, 0);
    }

    // Extract residual norms from logger
    scalar initRes = 0.0;
    scalar finalRes = 0.0;
    int numIter = 0;
    if (residualLogger->has_converged())
    {
        auto resNorm = gko::as<gko::matrix::Dense<scalar>>(residualLogger->get_residual_norm());
        finalRes = resNorm->at(0, 0);
        numIter = static_cast<int>(residualLogger->get_num_iterations());
    }

    return SolverStats(numIter, initRes, finalRes, solveTime);
}

} // namespace NeoN::la

#endif // NF_WITH_GINKGO
#endif // NF_WITH_MPI_SUPPORT
