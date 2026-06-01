// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO
#ifdef NF_WITH_MPI_SUPPORT

#include "NeoN/linearAlgebra/ginkgo.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/core/error.hpp"

#include <memory>
#include <vector>

#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::la::ginkgo
{

std::shared_ptr<gko::LinOp> gkoVecViewDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    scalar* ptr,
    localIdx s
)
{
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    auto size = static_cast<std::size_t>(s);
    return gko::share(dist_vec::create(
        exec,
        comm,
        vec::create(exec, gko::dim<2> {size, 1}, gko::array<scalar>::view(exec, size, ptr), 1)
    ));
}

std::shared_ptr<const gko::LinOp> gkoConstVecViewDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const scalar* ptr,
    localIdx s
)
{
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    auto size = static_cast<std::size_t>(s);
    return gko::share(dist_vec::create_const(
        exec,
        comm,
        vec::create_const(
            exec, gko::dim<2> {size, 1}, gko::array<scalar>::const_view(exec, size, ptr), 1
        )
    ));
}
template<typename IndexType>
std::shared_ptr<const gko::LinOp> createGkoMtxDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const CSRMatrix<scalar, IndexType>& mtx,
    const COOMatrix<scalar, IndexType>& bmtx,
    const CommunicationPattern& commPattern
)
{
    using global_index_type = gko::int64;
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, global_index_type>;

    auto vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(mtx.values().size()), mtx.values().data()
    );
    auto col = gko::array<IndexType>::const_view(
        exec,
        static_cast<gko::size_type>(mtx.sparsity()->colIdxs().size()),
        mtx.sparsity()->colIdxs().data()
    );
    auto row = gko::array<IndexType>::const_view(
        exec,
        static_cast<gko::size_type>(mtx.sparsity()->rowOffs().size()),
        mtx.sparsity()->rowOffs().data()
    );

    auto nrows = static_cast<gko::size_type>(mtx.sparsity()->rows());

    auto partition = gko::share(
        gko::experimental::distributed::build_partition_from_local_size<label, global_index_type>(
            exec, comm, nrows
        )
    );

    std::shared_ptr<const gko::LinOp> localMtx =
        gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
            exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
        ));

    // Non-local columns hold the global neighbour cell indices (stored in bmtx colIdxs).
    // gko::experimental::distributed::index_map (and map_to_local) require them as
    // global_index_type (int64). The off-diagonal matrix stores indices as IndexType, so the
    // widening happens here on the executor (no host round-trip); when IndexType already equals
    // global_index_type it is a straight copy. The same array serves both as the index_map's
    // recv_connections and as the columns mapped to local non-local indices.
    const auto nNonLocalNnz = static_cast<gko::size_type>(bmtx.values().size());

    gko::array<global_index_type> globalCols {exec, nNonLocalNnz};
    {
        auto* dst = globalCols.get_data();
        auto src = bmtx.sparsity()->colIdxs().view();
        parallelFor(
            bmtx.exec(),
            {0, static_cast<localIdx>(nNonLocalNnz)},
            KOKKOS_LAMBDA(const localIdx i) { dst[i] = static_cast<global_index_type>(src[i]); },
            "widenOffDiagonalColumns"
        );
        // ensure the widened columns are written before Ginkgo consumes the buffer
        fence(bmtx.exec());
    }

    auto imap = gko::experimental::distributed::index_map<label, global_index_type>(
        exec, partition, comm.rank(), globalCols
    );

    // numNonLocalElements: unique remote columns (used as the COO column dimension).
    // nNonLocalNnz: one entry per processor face (used as the COO nnz).
    // These differ when a local cell couples to the same remote cell via >1 face.
    const auto numNonLocalElements = imap.get_non_local_size();

    auto non_loc_vals = gko::array<scalar>::const_view(exec, nNonLocalNnz, bmtx.values().data());

    // rowIdxs() already holds local row indices: the global offset is no longer applied during
    // matrix creation on the NeoN side, so the row indices can be viewed directly without a host
    // round-trip or an offset subtraction.
    auto non_loc_row =
        gko::array<IndexType>::const_view(exec, nNonLocalNnz, bmtx.sparsity()->rowIdxs().data());

    // imap deduplicates: multiple faces to the same remote cell map to the same local column.
    auto comp_non_loc_col =
        imap.map_to_local(globalCols, gko::experimental::distributed::index_space::non_local);

    // The non-local (off-diagonal) COO must be sorted by row for Ginkgo's CUDA Coo::apply2: its
    // warp segmented-scan incorrectly reduces non-contiguous same-row entries otherwise, making the
    // GPU non-local apply non-symmetric (the distributed pressure CG then stalls/diverges, worse
    // with more processor faces). The Reference/CPU apply2 is order-robust. NeoN builds the
    // off-diagonal in processor-face order; the row-sort permutation is precomputed once when the
    // off-diagonal sparsity is created (CommunicationPattern::offDiagRowSortPerm), so here we only
    // gather (row, col, value) through it — no per-build sort. Row-sorting is a no-op for the
    // matrix (a COO is an unordered set of entries).
    const auto& offDiagRowSortPerm = commPattern.offDiagRowSortPerm;
    NF_ASSERT(
        offDiagRowSortPerm.size() == nNonLocalNnz,
        "offDiagRowSortPerm size mismatch: expected one entry per processor face"
    );
    gko::array<scalar> sortedVals {exec, nNonLocalNnz};
    gko::array<IndexType> sortedCol {exec, nNonLocalNnz};
    gko::array<IndexType> sortedRow {exec, nNonLocalNnz};
    {
        auto host = exec->get_master();
        auto rowH = non_loc_row;
        rowH.set_executor(host);
        auto colH = comp_non_loc_col;
        colH.set_executor(host);
        auto valH = non_loc_vals.copy_to_array();
        valH.set_executor(host);
        gko::array<scalar> vS {host, nNonLocalNnz};
        gko::array<IndexType> cS {host, nNonLocalNnz};
        gko::array<IndexType> rS {host, nNonLocalNnz};
        const auto* rp = rowH.get_const_data();
        const auto* cp = colH.get_const_data();
        const auto* vp = valH.get_const_data();
        for (gko::size_type i = 0; i < nNonLocalNnz; ++i)
        {
            const auto src = static_cast<gko::size_type>(offDiagRowSortPerm[i]);
            rS.get_data()[i] = rp[src];
            cS.get_data()[i] = static_cast<IndexType>(cp[src]);
            vS.get_data()[i] = vp[src];
        }
        vS.set_executor(exec);
        cS.set_executor(exec);
        rS.set_executor(exec);
        sortedVals = std::move(vS);
        sortedCol = std::move(cS);
        sortedRow = std::move(rS);
    }

    auto nonLocalMtx = gko::share(gko::matrix::Coo<scalar, IndexType>::create(
        exec,
        gko::dim<2> {nrows, numNonLocalElements},
        std::move(sortedVals),
        std::move(sortedCol),
        std::move(sortedRow)
    ));

    return gko::share(dist_mtx::create(
        exec, comm, imap, std::const_pointer_cast<gko::LinOp>(localMtx), nonLocalMtx
    ));
}

SolverStatsEntry solve_impl_dist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const Vector<scalar>& rhs,
    Vector<scalar>& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    std::unique_ptr<gko::LinOp> solver,
    const L1ResidualControl* l1Control = nullptr
)
{
    exec->synchronize();
    auto startEval = std::chrono::steady_clock::now();
    using vec = gko::matrix::Dense<scalar>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    label nrows = rhs.size();

    const auto b = gkoConstVecViewDist(exec, comm, rhs.data(), nrows);
    auto x = gkoVecViewDist(exec, comm, xIn.data(), nrows);

    // L1-scaled residual path: stop and report on the (globally reduced) scaled residual.
    if (l1Control != nullptr)
    {
        auto l1Res = solveWithL1StopDist(
            exec,
            mtx,
            std::dynamic_pointer_cast<const dist_vec>(b),
            std::dynamic_pointer_cast<dist_vec>(x),
            solver.get(),
            *l1Control
        );
        exec->synchronize();
        auto endEval = std::chrono::steady_clock::now();
        auto duration =
            static_cast<scalar>(
                std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
            )
            / 1000.0;
        return {static_cast<label>(l1Res.numIter), l1Res.initResNorm, l1Res.finalResNorm, duration};
    }

    auto rhsCopy = Vector<scalar>(rhs);
    auto res = gkoVecViewDist(exec, comm, rhsCopy.data(), nrows);

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    mtx->apply(one, x, neg_one, res);

    auto init = gko::initialize<vec>({0.0}, exec);
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    gko::as<dist_vec>(res)->compute_norm2(init);
    scalar initResNorm = retrieve(init);

    std::shared_ptr<const gko::log::Convergence<scalar>> logger =
        gko::log::Convergence<scalar>::create();
    solver->add_logger(logger);
    solver->apply(b, x);

    auto rhsCopyFinal = Vector<scalar>(rhs);
    auto resFinal = gkoVecViewDist(exec, comm, rhsCopyFinal.data(), nrows);
    mtx->apply(one, x, neg_one, resFinal);
    auto finalNormVec = gko::initialize<vec>({0.0}, exec);
    gko::as<dist_vec>(resFinal)->compute_norm2(finalNormVec);
    scalar finalResNorm = retrieve(finalNormVec);

    auto numIter = label(logger->get_num_iterations());
    exec->synchronize();
    auto endEval = std::chrono::steady_clock::now();
    auto duration =
        static_cast<scalar>(
            std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
        )
        / 1000.0;

    return {numIter, initResNorm, finalResNorm, duration};
}

template<unsigned int I>
void solveComponentDist(
    auto& sys, auto& x, auto& exec, auto& factory, auto& stats, const L1ResidualControl* l1Control
)
{
    auto rhs = getComponent<I>(sys.rhs());
    auto xcopy = getComponent<I>(x);
    auto values = getComponent<I>(sys.matrix().values());
    auto sparsity = sys.matrix().sparsity();
    auto mtx = CSRMatrix<scalar, localIdx> {values, sparsity};

    auto nonLocalValues = getComponent<I>(sys.offDiagonalMatrix().values());
    auto nonLocalSparsity = sys.offDiagonalMatrix().sparsity();
    auto nonLocalMtx = COOMatrix<scalar, localIdx> {nonLocalValues, nonLocalSparsity};

    const CommunicationPattern& commPattern = sys.commPattern();
    auto comm = gko::experimental::mpi::communicator(
        commPattern.env.comm(), !commPattern.env.gpuAwareMpi()
    );
    auto gkoMtx = createGkoMtxDist(exec, comm, mtx, nonLocalMtx, commPattern);
    auto solver = factory->generate(gkoMtx);
    stats.entries.push_back(
        solve_impl_dist(exec, comm, rhs, xcopy, gkoMtx, std::move(solver), l1Control)
    );
    setComponent<I>(xcopy, x);
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
) const
{
    const CommunicationPattern& commPattern = sys.commPattern();
    auto comm = gko::experimental::mpi::communicator(
        commPattern.env.comm(), !commPattern.env.gpuAwareMpi()
    );
    auto gkoMtx =
        createGkoMtxDist(gkoExec_, comm, sys.matrix(), sys.offDiagonalMatrix(), commPattern);
    auto solver = factory_->generate(gkoMtx);
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    return {solve_impl_dist(gkoExec_, comm, sys.rhs(), x, gkoMtx, std::move(solver), l1Control)};
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
) const
{
    auto stats = SolverStats {};
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    solveComponentDist<0>(sys, x, gkoExec_, factory_, stats, l1Control);
    solveComponentDist<1>(sys, x, gkoExec_, factory_, stats, l1Control);
    solveComponentDist<2>(sys, x, gkoExec_, factory_, stats, l1Control);
    return stats;
}

// Solve one Vec3 rhs component against a shared distributed scalar matrix (segregated form).
template<unsigned int I>
void solveVec3RhsComponentDist(
    const Vector<Vec3>& rhs,
    Vector<Vec3>& x,
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    std::shared_ptr<const gko::LinOpFactory> factory,
    std::shared_ptr<const gko::LinOp> gkoMtx,
    SolverStats& stats,
    const L1ResidualControl* l1Control
)
{
    auto rhsComp = getComponent<I>(rhs);
    auto xcopy = getComponent<I>(x);
    auto solver = factory->generate(gkoMtx);
    stats.entries.push_back(
        solve_impl_dist(exec, comm, rhsComp, xcopy, gkoMtx, std::move(solver), l1Control)
    );
    setComponent<I>(xcopy, x);
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<scalar, Vec3, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>>& sys,
    Vector<Vec3>& x
) const
{
    const CommunicationPattern& commPattern = sys.commPattern();
    auto comm = gko::experimental::mpi::communicator(
        commPattern.env.comm(), !commPattern.env.gpuAwareMpi()
    );
    // The matrix is already scalar and shared across all three components, so build the
    // distributed operator once and reuse it for each Vec3 rhs component.
    auto gkoMtx =
        createGkoMtxDist(gkoExec_, comm, sys.matrix(), sys.offDiagonalMatrix(), commPattern);

    // TODO here Ginkgos multiple RHS solver could be used
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    auto stats = SolverStats {};
    solveVec3RhsComponentDist<0>(sys.rhs(), x, gkoExec_, comm, factory_, gkoMtx, stats, l1Control);
    solveVec3RhsComponentDist<1>(sys.rhs(), x, gkoExec_, comm, factory_, gkoMtx, stats, l1Control);
    solveVec3RhsComponentDist<2>(sys.rhs(), x, gkoExec_, comm, factory_, gkoMtx, stats, l1Control);
    return stats;
}

}

#endif // NF_WITH_MPI_SUPPORT
#endif // NF_WITH_GINKGO
