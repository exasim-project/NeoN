// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// Guarded like ginkgo.hpp: the generated NeoN.hpp umbrella header includes every header
// unconditionally, so this must be a no-op in a build configured without Ginkgo.
#if NF_WITH_GINKGO

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
 *  - **local** (localized/Schwarz MG, each rank owns a plain Csr): compose the per-level
 *    RowGatherer injections into a single injection Csr; the last inner Pgm's coarse op is
 *    A_merged.
 *  - **distributed** (global MG, fine op is a gko::experimental::distributed::Matrix): the
 *    distributed Pgm prolong/restrict are block-diagonal across ranks (empty off-diagonal
 *    block; only the coarse op carries cross-rank coupling -- see Ginkgo pgm.cpp
 *    distributed_setup). So the composition is rank-local: compose each prolong's DIAG-block
 *    RowGatherer exactly as in the local path, then re-wrap the merged local injection as a
 *    block-diagonal distributed::Matrix. A_merged is the last inner Pgm's distributed coarse
 *    op (already carries the correct off-diagonal coupling + coarse index map), used directly.
 *
 * The factory supports LinOpFactory::generate_reuse() (see reuse_data_type), so a
 * gko::solver::Multigrid generated through generate_reuse() keeps the merged aggregation and only
 * recomputes the coarse operators.
 */
template<typename ValueType = gko::default_precision, typename IndexType = gko::int32>
class MergedPgm : public gko::LinOp, public gko::multigrid::EnableMultigridLevel<ValueType>
{
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
    class reuse_data_type;
    GKO_ENABLE_LIN_OP_FACTORY_WITH_REUSE(MergedPgm, parameters, Factory, reuse_data_type);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Records, on the first generate_reuse(), the reuse data of every inner Pgm and the merged
     * prolong/restrict. Later generate_reuse() calls run each inner Pgm through its own
     * generate_reuse() -- aggregates frozen, only the coarse values recomputed (no SpGEMM) -- so
     * the composed aggregation, and with it the merged injection, stays valid and is reused as is.
     */
    class reuse_data_type : public gko::LinOpFactory::ReuseData
    {
        friend class MergedPgm;

        bool initialized_ = false;
        bool distributed_ = false;
        gko::dim<2> size_ {};
        std::vector<std::unique_ptr<gko::LinOpFactory::ReuseData>> levels_;
        std::shared_ptr<const gko::LinOp> prolong_;
        std::shared_ptr<const gko::LinOp> restrict_;
    };

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

    explicit MergedPgm(std::shared_ptr<const gko::Executor> exec) : gko::LinOp(std::move(exec)) {}

    MergedPgm(const Factory* factory, std::shared_ptr<const gko::LinOp> system_matrix)
        : MergedPgm(factory, std::move(system_matrix), nullptr)
    {}

    MergedPgm(
        const Factory* factory,
        std::shared_ptr<const gko::LinOp> system_matrix,
        reuse_data_type& reuse_data
    )
        : MergedPgm(factory, std::move(system_matrix), &reuse_data)
    {}

    MergedPgm(
        const Factory* factory,
        std::shared_ptr<const gko::LinOp> system_matrix,
        reuse_data_type* reuse_data
    )
        : gko::LinOp(factory->get_executor(), system_matrix->get_size()),
          gko::multigrid::EnableMultigridLevel<ValueType>(system_matrix),
          parameters_ {factory->get_parameters()}, system_matrix_ {system_matrix}
    {
        GKO_ASSERT(parameters_.merge_levels >= 1u);
        if (system_matrix_->get_size()[0] != 0)
        {
            if (is_distributed(system_matrix_.get()))
            {
                generateDistributed(reuse_data);
            }
            else
            {
                generateLocal(reuse_data);
            }
        }
    }

    /** Throws if `input` can't be used with `reuse_data`. */
    static void check_reuse_consistent(
        const Factory*, const gko::LinOp* input, const reuse_data_type& reuse_data
    )
    {
        if (!reuse_data.initialized_) return;
        GKO_ASSERT_EQUAL_DIMENSIONS(input, reuse_data.size_);
        if (is_distributed(input) != reuse_data.distributed_)
        {
            GKO_INVALID_STATE(
                "generate_reuse needs a distributed matrix exactly if the reuse data was "
                "initialized with one"
            );
        }
    }

    // ---- localized / Schwarz path: fine op is a plain Csr on this rank ----
    void generateLocal(reuse_data_type* reuseData)
    {
        auto A = as_csr(system_matrix_);
        const auto fineRows = A->get_size()[0];
        const bool reuseMerged = reuseData && reuseData->initialized_;

        // Run Pgm `merge_levels` times, composing the per-level aggregations BY INDEX. Each inner
        // Pgm coarsens `coarse` and yields a piecewise-constant prolong (a RowGatherer: fine row ->
        // aggregate) plus the next coarse operator. Because the prolongs are injections, the merged
        // prolongation is the injection of the composed aggregate map `mergedAgg[i] =
        // aggThis[mergedAgg[i]]`
        // -- no Csr*Csr SpGEMM (cuSPARSE aborts with INSUFFICIENT_RESOURCES on large operands).
        // When reusing, the inner aggregates are frozen, so the recorded merged injection is reused
        // and only the coarse operator is refreshed.
        auto pgmFactory = makePgm();
        std::vector<std::unique_ptr<gko::LinOpFactory::ReuseData>> newSlots;
        std::shared_ptr<const csr> coarse = A;
        std::vector<IndexType> mergedAgg; // fine row -> current coarse index (host)
        gko::size_type coarseRows = 0;
        for (unsigned i = 0; i < parameters_.merge_levels; ++i)
        {
            auto level = generateLevel(pgmFactory.get(), coarse, i, reuseData, newSlots);
            if (!reuseMerged)
            {
                auto aggThis = gather_agg(level.get()); // host, length coarse->rows
                coarseRows = level->get_prolong_op()->get_size()[1];
                composeAgg(mergedAgg, aggThis, fineRows, i);
            }
            coarse = as_csr(level->get_coarse_op());
        }

        if (reuseMerged)
        {
            prolong_ = reuseData->prolong_;
            restrict_ = reuseData->restrict_;
        }
        else
        {
            auto pCsr = make_injection(mergedAgg, fineRows, coarseRows); // Csr, fine x coarse
            prolong_ = pCsr;
            restrict_ = gko::share(gko::as<csr>(pCsr->transpose())); // Csr, coarse x fine
        }
        // coarse == P_merged^T A P_merged already (last Pgm level's coarse op); use it directly.
        this->set_multigrid_level(prolong_, coarse, restrict_);
        recordReuse(reuseData, std::move(newSlots), false);
    }

#ifdef NF_WITH_MPI_SUPPORT
    // ---- global path: fine op is a distributed::Matrix ----
    void generateDistributed(reuse_data_type* reuseData)
    {
        auto exec = this->get_executor();
        auto distFine = gko::as<const dist_mtx>(system_matrix_);
        auto comm = gko::as<const gko::experimental::distributed::DistributedBase>(system_matrix_)
                        ->get_communicator();
        const auto fineGlobalRows = system_matrix_->get_size()[0];
        // The prolong composition is rank-local because the distributed Pgm prolong is
        // block-diagonal; all sizes below are LOCAL (this rank's diag block).
        const auto localFineRows = distFine->get_diag_matrix()->get_size()[0];
        const bool reuseMerged = reuseData && reuseData->initialized_;

        auto pgmFactory = makePgm();
        std::vector<std::unique_ptr<gko::LinOpFactory::ReuseData>> newSlots;
        std::shared_ptr<const gko::LinOp> coarse = system_matrix_; // stays distributed
        std::vector<IndexType> mergedAgg; // LOCAL fine row -> LOCAL coarse idx
        gko::size_type localCoarseRows = 0;
        for (unsigned i = 0; i < parameters_.merge_levels; ++i)
        {
            auto level = generateLevel(pgmFactory.get(), coarse, i, reuseData, newSlots);
            if (!reuseMerged)
            {
                // Aggregation lives in the block-diagonal prolong's DIAG block (a RowGatherer).
                auto prolongDiag =
                    gko::as<const dist_mtx>(level->get_prolong_op())->get_diag_matrix();
                auto aggThis =
                    agg_from_rowgatherer(prolongDiag.get()); // host, length = local fine rows
                // local coarse rows == this level's coarse DIAG block rows
                localCoarseRows = gko::as<const dist_mtx>(level->get_coarse_op())
                                      ->get_diag_matrix()
                                      ->get_size()[0];
                composeAgg(mergedAgg, aggThis, localFineRows, i);
            }
            coarse = level->get_coarse_op(); // distributed coarse op -> next Pgm fine op
        }

        if (reuseMerged)
        {
            prolong_ = reuseData->prolong_;
            restrict_ = reuseData->restrict_;
        }
        else
        {
            const auto coarseGlobalRows = coarse->get_size()[0];
            // Build the merged local injection Csr, then re-wrap prolong/restrict as
            // block-diagonal distributed matrices (diag block only, empty off-diagonal) -- mirrors
            // pgm.cpp's distributed_setup, which builds prolong/restrict from the local block
            // alone.
            std::shared_ptr<csr> pLocal = make_injection(mergedAgg, localFineRows, localCoarseRows);
            std::shared_ptr<gko::LinOp> rLocal = gko::share(gko::as<csr>(pLocal->transpose()));
            prolong_ = gko::share(
                dist_mtx::create(exec, comm, gko::dim<2> {fineGlobalRows, coarseGlobalRows}, pLocal)
            );
            restrict_ = gko::share(
                dist_mtx::create(exec, comm, gko::dim<2> {coarseGlobalRows, fineGlobalRows}, rLocal)
            );
        }
        // coarse == the last inner Pgm's distributed coarse op == A_merged; use it directly.
        this->set_multigrid_level(prolong_, coarse, restrict_);
        recordReuse(reuseData, std::move(newSlots), true);
    }
#else
    void generateDistributed(reuse_data_type*) { GKO_NOT_IMPLEMENTED; }
#endif

private:

    // Generate inner Pgm level `i`: plainly without reuse data, through the recorded slot once the
    // reuse data is initialized, and otherwise through a fresh slot that recordReuse() commits.
    static std::unique_ptr<pgm> generateLevel(
        const typename pgm::Factory* pgmFactory,
        std::shared_ptr<const gko::LinOp> fine,
        unsigned i,
        reuse_data_type* reuseData,
        std::vector<std::unique_ptr<gko::LinOpFactory::ReuseData>>& newSlots
    )
    {
        if (!reuseData) return pgmFactory->generate(std::move(fine));
        if (reuseData->initialized_)
        {
            return pgmFactory->generate_reuse(std::move(fine), *reuseData->levels_.at(i));
        }
        newSlots.push_back(pgmFactory->create_empty_reuse_data());
        return pgmFactory->generate_reuse(std::move(fine), *newSlots.back());
    }

    // Fill an empty reuse data only once the whole level was generated, so that a throw leaves it
    // empty (LinOpFactory::generate_reuse_impl guarantee).
    void recordReuse(
        reuse_data_type* reuseData,
        std::vector<std::unique_ptr<gko::LinOpFactory::ReuseData>> newSlots,
        bool distributed
    ) const
    {
        if (!reuseData || reuseData->initialized_) return;
        reuseData->levels_ = std::move(newSlots);
        reuseData->prolong_ = prolong_;
        reuseData->restrict_ = restrict_;
        reuseData->size_ = system_matrix_->get_size();
        reuseData->distributed_ = distributed;
        reuseData->initialized_ = true;
    }

    static bool is_distributed(const gko::LinOp* op)
    {
#ifdef NF_WITH_MPI_SUPPORT
        return dynamic_cast<const gko::experimental::distributed::DistributedBase*>(op) != nullptr;
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
    // the local and distributed paths -- the distributed prolong is block-diagonal so its
    // diag-block aggregation composes rank-locally, exactly like a local Csr aggregation.
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

    // Pull a RowGatherer's per-row aggregate map (fine row i -> its aggregate) to the host. Length
    // == the gatherer's row count; values in [0, coarseRows). One entry per row.
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

    // NB: `parameters_` is provided by GKO_ENABLE_LIN_OP_FACTORY_WITH_REUSE; do not redeclare it.
    std::shared_ptr<const gko::LinOp> system_matrix_;
    // Merged prolong/restrict: a Csr in the local path, a block-diagonal distributed::Matrix in the
    // global path. Structural (composed once); shared with the reuse data.
    std::shared_ptr<const gko::LinOp> prolong_;
    std::shared_ptr<const gko::LinOp> restrict_;
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

#endif // NF_WITH_GINKGO
