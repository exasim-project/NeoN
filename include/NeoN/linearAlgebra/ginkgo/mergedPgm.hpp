// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/solver/multigrid.hpp>

#ifdef NF_WITH_MPI_SUPPORT
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#endif

namespace NeoN::la::ginkgo
{

/**
 * @brief `mergeLevels`-style coarsening: Pgm run `merge_levels` times, prolongations
 * composed BY INDEX into one merged multigrid level.
 *
 * merge_levels == 1 reproduces plain Pgm (single coarsening step). merge_levels == k builds a level
 * that jumps k Pgm steps at once; the coarsest grid is unchanged, only the number of intermediate
 * levels drops (~depth/k).
 *
 * Two paths, selected at generate() time by the runtime type of the fine operator:
 *  - **local** (localized/Schwarz MG, each rank owns a plain Csr): compose the per-level RowGatherer
 *    injections into a single injection Csr; the last inner Pgm's coarse op is A_merged.
 *  - **distributed** (global MG, fine op is a gko::experimental::distributed::Matrix): the distributed
 *    Pgm prolong/restrict are block-diagonal across ranks (empty off-diagonal block; only the coarse
 *    op carries cross-rank coupling -- see Ginkgo pgm.cpp distributed_setup). So the composition is
 *    rank-local: compose each prolong's DIAG-block RowGatherer exactly as in the local path, then
 *    re-wrap the merged local injection as a block-diagonal distributed::Matrix. A_merged is the last
 *    inner Pgm's distributed coarse op (already carries the correct off-diagonal coupling + coarse
 *    index map), used directly.
 */
template<typename ValueType = gko::default_precision, typename IndexType = gko::int32>
class MergedPgm :
    public gko::EnableLinOp<MergedPgm<ValueType, IndexType>>,
    public gko::multigrid::EnableMultigridLevel<ValueType>,
    public gko::UpdateMatrixValue
{
    friend class gko::EnableLinOp<MergedPgm>;
    friend class gko::EnablePolymorphicObject<MergedPgm, gko::LinOp>;

public:

    using value_type = ValueType;
    using index_type = IndexType;
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using pgm = gko::multigrid::Pgm<ValueType, IndexType>;
    using row_gatherer = gko::matrix::RowGatherer<IndexType>;
#ifdef NF_WITH_MPI_SUPPORT
    // NeoN always builds its distributed system matrix with a 64-bit global index type
    // (@see createGkoMtxDist in ginkgoDistributed.cpp); Pgm keeps the same global index type for
    // every coarser level, so all distributed operators seen here are Matrix<Value, Index, int64>.
    using dist_mtx = gko::experimental::distributed::Matrix<ValueType, IndexType, gko::int64>;
#endif

    std::shared_ptr<const gko::LinOp> get_system_matrix() const { return system_matrix_; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /** Number of Pgm coarsening steps merged into one level (OpenFOAM `mergeLevels`). */
        unsigned GKO_FACTORY_PARAMETER_SCALAR(merge_levels, 2u);
        /** Forwarded to each inner Pgm (see gko::multigrid::Pgm). */
        unsigned GKO_FACTORY_PARAMETER_SCALAR(max_iterations, 15u);
        double GKO_FACTORY_PARAMETER_SCALAR(max_unassigned_ratio, 0.05);
        bool GKO_FACTORY_PARAMETER_SCALAR(deterministic, false);
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(MergedPgm, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    void update_matrix_value(std::shared_ptr<const gko::LinOp> new_matrix) override
    {
        system_matrix_ = new_matrix;
        // Value-only refresh with FROZEN structure, cuSPARSE-free. The merged prolongation
        // (prolong_/restrict_) is structural and stays fixed; only the coarse operator's values
        // change. Rather than recompute A_merged = R A P by SpGEMM (cuSPARSE SpGEMM aborts with
        // CUSPARSE_STATUS_INSUFFICIENT_RESOURCES on these large injection operands), walk the
        // retained Pgm chain: each Pgm::update_matrix_value scatters the new fine values into its
        // frozen coarse structure (no SpGEMM) and refreshes get_coarse_op() in place; that refreshed
        // coarse feeds the next level's fine op. After the last level, get_coarse_op() is the
        // refreshed A_merged. This keeps the cached-solver (update_matrix_value) reuse engaged.
        const bool isDist = is_distributed(new_matrix);
        // local path converts to Csr; distributed path feeds the distributed op through unchanged
        // (Pgm::update_matrix_value has its own distributed branch).
        std::shared_ptr<const gko::LinOp> fineOp = isDist ? new_matrix : as_csr(new_matrix);
        for (auto& lvl : levels_)
        {
            gko::as<gko::UpdateMatrixValue>(lvl.get())->update_matrix_value(fineOp);
            fineOp = gko::as<gko::multigrid::MultigridLevel>(lvl.get())->get_coarse_op();
        }
        std::shared_ptr<const gko::LinOp> coarse =
            gko::as<gko::multigrid::MultigridLevel>(levels_.back().get())->get_coarse_op();
        if (!isDist) coarse = as_csr(coarse);
        this->set_multigrid_level(prolong_, coarse, restrict_);
    }

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        this->get_composition()->apply(b, x);
    }
    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        this->get_composition()->apply(alpha, b, beta, x);
    }

    explicit MergedPgm(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<MergedPgm>(std::move(exec))
    {}

    MergedPgm(const Factory* factory, gko::LinOpGenerateComponents components)
        : gko::EnableLinOp<MergedPgm>(
            factory->get_executor(), components.system_matrix->get_size()
        ),
          gko::multigrid::EnableMultigridLevel<ValueType>(components.system_matrix),
          parameters_ {factory->get_parameters()}, system_matrix_ {components.system_matrix}
    {
        GKO_ASSERT(parameters_.merge_levels >= 1u);
        if (system_matrix_->get_size()[0] != 0)
        {
            if (is_distributed(system_matrix_))
            {
                generateDistributed();
            }
            else
            {
                generateLocal();
            }
        }
    }

    // ---- localized / Schwarz path: fine op is a plain Csr on this rank -----------------------------
    void generateLocal()
    {
        auto exec = this->get_executor();
        auto A = as_csr(system_matrix_);
        const auto fineRows = A->get_size()[0];

        // Run Pgm `merge_levels` times, composing the per-level aggregations BY INDEX. Each inner
        // Pgm coarsens `coarse` and yields a piecewise-constant prolong (a RowGatherer: fine row ->
        // aggregate) plus the next coarse operator. Because the prolongs are injections, the merged
        // prolongation is the injection of the composed aggregate map `mergedAgg[i] =
        // aggThis[mergedAgg[i]]`
        // -- no Csr*Csr SpGEMM (cuSPARSE aborts with INSUFFICIENT_RESOURCES on large operands). The
        // Pgm levels are retained so update_matrix_value can refresh the coarse values in place.
        levels_.clear();
        std::shared_ptr<const csr> coarse = A;
        std::vector<IndexType> mergedAgg; // fine row -> current coarse index (host)
        gko::size_type coarseRows = 0;
        for (unsigned i = 0; i < parameters_.merge_levels; ++i)
        {
            auto level = makePgm()->generate(coarse);
            auto aggThis = gather_agg(level.get()); // host, length coarse->rows, in [0, coarseRows)
            coarseRows = level->get_prolong_op()->get_size()[1];
            composeAgg(mergedAgg, aggThis, fineRows, i);
            coarse = as_csr(level->get_coarse_op());
            levels_.push_back(gko::share(std::move(level)));
        }

        auto pCsr = make_injection(mergedAgg, fineRows, coarseRows); // Csr, fine x coarse
        prolong_ = pCsr;
        restrict_ = gko::share(gko::as<csr>(pCsr->transpose())); // Csr, coarse x fine
        // coarse == P_merged^T A P_merged already (last Pgm level's coarse op); use it directly.
        this->set_multigrid_level(prolong_, coarse, restrict_);
    }

#ifdef NF_WITH_MPI_SUPPORT
    // ---- global path: fine op is a distributed::Matrix ---------------------------------------------
    void generateDistributed()
    {
        auto exec = this->get_executor();
        auto distFine = gko::as<const dist_mtx>(system_matrix_);
        auto comm = gko::as<const gko::experimental::distributed::DistributedBase>(system_matrix_)
                        ->get_communicator();
        const auto fineGlobalRows = system_matrix_->get_size()[0];
        // The prolong composition is rank-local because the distributed Pgm prolong is
        // block-diagonal; all sizes below are LOCAL (this rank's diag block).
        const auto localFineRows = distFine->get_diag_matrix()->get_size()[0];

        levels_.clear();
        std::shared_ptr<const gko::LinOp> coarse = system_matrix_; // stays distributed
        std::vector<IndexType> mergedAgg;                          // LOCAL fine row -> LOCAL coarse idx
        gko::size_type localCoarseRows = 0;
        for (unsigned i = 0; i < parameters_.merge_levels; ++i)
        {
            auto level = makePgm()->generate(coarse); // distributed Pgm level
            // Aggregation lives in the block-diagonal prolong's DIAG block (a RowGatherer).
            auto prolongDiag = gko::as<const dist_mtx>(level->get_prolong_op())->get_diag_matrix();
            auto aggThis = agg_from_rowgatherer(prolongDiag.get()); // host, length = local fine rows
            // local coarse rows == this level's coarse DIAG block rows
            localCoarseRows =
                gko::as<const dist_mtx>(level->get_coarse_op())->get_diag_matrix()->get_size()[0];
            composeAgg(mergedAgg, aggThis, localFineRows, i);
            coarse = level->get_coarse_op(); // distributed coarse op -> next Pgm fine op
            levels_.push_back(gko::share(std::move(level)));
        }
        const auto coarseGlobalRows = coarse->get_size()[0];

        // Build the merged local injection Csr, then re-wrap prolong/restrict as block-diagonal
        // distributed matrices (diag block only, empty off-diagonal) -- mirrors pgm.cpp's
        // distributed_setup, which builds prolong/restrict from the local block alone.
        std::shared_ptr<csr> pLocal = make_injection(mergedAgg, localFineRows, localCoarseRows);
        std::shared_ptr<gko::LinOp> rLocal = gko::share(gko::as<csr>(pLocal->transpose()));
        prolong_ = gko::share(
            dist_mtx::create(exec, comm, gko::dim<2> {fineGlobalRows, coarseGlobalRows}, pLocal)
        );
        restrict_ = gko::share(
            dist_mtx::create(exec, comm, gko::dim<2> {coarseGlobalRows, fineGlobalRows}, rLocal)
        );
        // coarse == the last inner Pgm's distributed coarse op == A_merged; use it directly.
        this->set_multigrid_level(prolong_, coarse, restrict_);
    }
#else
    void generateDistributed() { GKO_NOT_IMPLEMENTED; }
#endif

private:

    static bool is_distributed(const std::shared_ptr<const gko::LinOp>& op)
    {
#ifdef NF_WITH_MPI_SUPPORT
        return std::dynamic_pointer_cast<const gko::experimental::distributed::DistributedBase>(op)
               != nullptr;
#else
        static_cast<void>(op);
        return false;
#endif
    }

    std::unique_ptr<typename pgm::Factory> makePgm() const
    {
        return pgm::build()
            .with_max_iterations(parameters_.max_iterations)
            .with_max_unassigned_ratio(parameters_.max_unassigned_ratio)
            .with_deterministic(parameters_.deterministic)
            .with_skip_sorting(parameters_.skip_sorting)
            .on(this->get_executor());
    }

    // mergedAgg[k] <- aggThis[mergedAgg[k]] (or aggThis directly on the first level). Identical for
    // the local and distributed paths -- the distributed prolong is block-diagonal so its diag-block
    // aggregation composes rank-locally, exactly like a local Csr aggregation.
    static void composeAgg(
        std::vector<IndexType>& mergedAgg,
        const gko::array<IndexType>& aggThis,
        gko::size_type fineRows,
        unsigned i
    )
    {
        if (i == 0)
        {
            mergedAgg.assign(aggThis.get_const_data(), aggThis.get_const_data() + fineRows);
        }
        else
        {
            for (gko::size_type k = 0; k < fineRows; ++k)
            {
                mergedAgg[k] = aggThis.get_const_data()[mergedAgg[k]];
            }
        }
    }

    static std::shared_ptr<const csr> as_csr(std::shared_ptr<const gko::LinOp> op)
    {
        if (auto c = std::dynamic_pointer_cast<const csr>(op)) return c;
        auto exec = op->get_executor();
        auto out = csr::create(exec);
        gko::as<const gko::ConvertibleTo<csr>>(op.get())->convert_to(out);
        return std::move(out);
    }

    // Pull a RowGatherer's per-row aggregate map (fine row i -> its aggregate) to the host. Length ==
    // the gatherer's row count; values in [0, coarseRows). One entry per row.
    gko::array<IndexType> agg_from_rowgatherer(const gko::LinOp* rgOp) const
    {
        auto exec = this->get_executor();
        auto host = exec->get_master();
        auto rg = gko::as<const row_gatherer>(rgOp);
        auto fineRows = rg->get_size()[0];
        gko::array<IndexType> agg(host, fineRows);
        host->copy_from(exec, fineRows, rg->get_const_row_idxs(), agg.get_data());
        return agg;
    }

    // Localized-path convenience: the level's prolong IS a RowGatherer.
    gko::array<IndexType> gather_agg(const gko::multigrid::MultigridLevel* level) const
    {
        return agg_from_rowgatherer(level->get_prolong_op().get());
    }

    // Build the merged injection Csr P (P[i, mergedAgg[i]] = 1) from the composed aggregate map on
    // the host, then clone to the device executor. row_ptrs = iota (one entry per row); values =
    // ones. One entry per row => naturally sorted. Returned non-const so it can serve as the
    // (mutable) diag block of a distributed::Matrix.
    std::shared_ptr<csr> make_injection(
        const std::vector<IndexType>& mergedAgg, gko::size_type fineRows, gko::size_type coarseRows
    ) const
    {
        auto exec = this->get_executor();
        auto host = exec->get_master();
        gko::array<IndexType> cols(host, fineRows);
        std::copy(mergedAgg.begin(), mergedAgg.end(), cols.get_data());
        gko::array<IndexType> rowPtrs(host, fineRows + 1);
        std::iota(rowPtrs.get_data(), rowPtrs.get_data() + fineRows + 1, IndexType {0});
        gko::array<ValueType> vals(host, fineRows);
        std::fill_n(vals.get_data(), fineRows, gko::one<ValueType>());

        auto pHost = csr::create(
            host,
            gko::dim<2> {fineRows, coarseRows},
            std::move(vals),
            std::move(cols),
            std::move(rowPtrs)
        );
        return gko::share(gko::clone(exec, pHost));
    }

    // NB: `parameters_` is provided by GKO_ENABLE_LIN_OP_FACTORY; do not redeclare it.
    std::shared_ptr<const gko::LinOp> system_matrix_;
    // Merged prolong/restrict: a Csr in the local path, a block-diagonal distributed::Matrix in the
    // global path. Structural (composed once); reused across update_matrix_value refreshes.
    std::shared_ptr<const gko::LinOp> prolong_;
    std::shared_ptr<const gko::LinOp> restrict_;
    // Retained inner Pgm chain (each a MultigridLevel + UpdateMatrixValue): lets
    // update_matrix_value refresh the merged coarse operator's values in place with frozen
    // aggregation, cuSPARSE-free.
    std::vector<std::shared_ptr<gko::LinOp>> levels_;
};

/**
 * @brief Build a named MergedPgm factory for the config registry (mirrors the L1-criterion
 *        registration in the GinkgoSolver ctor). Reference it from a configFile's `mg_level`, e.g.
 *        `"mg_level": ["neon::pgmMerge2"]`.
 */
// NB: return the CONCRETE factory type, not the abstract gko::LinOpFactory. gko::config::registry
// keys entries by base_type<T>::type, which is defined for concrete factories (-> LinOpFactory) but
// not for the abstract base — emplacing a shared_ptr<const LinOpFactory> fails to compile.
template<typename ValueType = gko::default_precision, typename IndexType = gko::int32>
inline std::shared_ptr<typename MergedPgm<ValueType, IndexType>::Factory> makeMergedPgmFactory(
    std::shared_ptr<const gko::Executor> exec, unsigned mergeLevels, bool deterministic = true
)
{
    return MergedPgm<ValueType, IndexType>::build()
        .with_merge_levels(mergeLevels)
        .with_deterministic(deterministic)
        .on(std::move(exec));
}

} // namespace NeoN::la::ginkgo
