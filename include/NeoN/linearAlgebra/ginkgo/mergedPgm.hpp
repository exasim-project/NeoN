// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// Guarded like ginkgo.hpp: the generated NeoN.hpp umbrella header includes every header
// unconditionally, so this must be a no-op in a build configured without Ginkgo.
#if NF_WITH_GINKGO

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/solver/multigrid.hpp>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#ifdef NF_WITH_MPI_SUPPORT
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#endif

namespace NeoN::la::ginkgo
{

namespace detail
{

/** @brief Residual a MergedPgm level last restricted, handed from its restriction to its
 * prolongation. gko::solver::Multigrid restricts r = b - A x right before descending and prolongs
 * the coarse correction right after returning, with r untouched in between, so the pointer is valid
 * for exactly that prolongation. */
struct ScaleCorrectionState
{
    const gko::LinOp* residual = nullptr;
};

/** @brief Whether the device executor @p exec runs on the Kokkos default execution space's stream.
 * NeoN creates its Ginkgo device executors there (createGkoExecutor); a Kokkos kernel is then
 * stream-ordered with the surrounding Ginkgo operations. Throws if @p exec is on another device,
 * whose memory a Kokkos kernel cannot access. SYCL queues are not compared.
 */
inline bool onKokkosStream([[maybe_unused]] const gko::Executor& exec)
{
#if defined(KOKKOS_ENABLE_CUDA)
    if (const auto* cuda = dynamic_cast<const gko::CudaExecutor*>(&exec))
    {
        NF_ASSERT(
            cuda->get_device_id() == Kokkos::device_id(),
            "MergedPgm: the Ginkgo executor is on another device than Kokkos"
        );
        return cuda->get_stream() == Kokkos::DefaultExecutionSpace().cuda_stream();
    }
#elif defined(KOKKOS_ENABLE_HIP)
    if (const auto* hip = dynamic_cast<const gko::HipExecutor*>(&exec))
    {
        NF_ASSERT(
            hip->get_device_id() == Kokkos::device_id(),
            "MergedPgm: the Ginkgo executor is on another device than Kokkos"
        );
        return hip->get_stream() == Kokkos::DefaultExecutionSpace().hip_stream();
    }
#endif
    return true;
}

/** @brief Run kernel(i) for i in [0, n) where @p exec's memory lives: as a Kokkos kernel for a
 * device (CUDA/HIP/SYCL) executor, a parallel host kernel for an OpenMP executor and a host loop
 * otherwise. A device executor on another stream than Kokkos' is synchronised around the kernel. */
template<typename Kernel>
void forEachOn(const gko::Executor& exec, gko::size_type n, const Kernel& kernel)
{
    // Host executors are their own master.
    if (exec.get_master().get() != &exec)
    {
        const bool sameStream = onKokkosStream(exec);
        if (!sameStream)
        {
            exec.synchronize();
        }
        parallelFor(GPUExecutor {}, {0, static_cast<localIdx>(n)}, kernel, "mergedPgm");
        if (!sameStream)
        {
            fence(GPUExecutor {});
        }
    }
    else if (dynamic_cast<const gko::OmpExecutor*>(&exec))
    {
        parallelFor(CPUExecutor {}, {0, static_cast<localIdx>(n)}, kernel, "mergedPgm");
    }
    else
    {
        for (localIdx i = 0; i < static_cast<localIdx>(n); ++i)
        {
            kernel(i);
        }
    }
}

/** @brief Rayleigh scale factors sf = num / den per column (1 where den == 0), computed on the
 * executor of the 1 x ncols operands instead of a host round trip. */
template<typename ValueType>
void rayleighScale(
    const gko::matrix::Dense<ValueType>* num,
    const gko::matrix::Dense<ValueType>* den,
    gko::matrix::Dense<ValueType>* sf
)
{
    const auto* n = num->get_const_values();
    const auto* d = den->get_const_values();
    auto* s = sf->get_values();
    forEachOn(
        *sf->get_executor(),
        sf->get_size()[1],
        NEON_LAMBDA(const localIdx c) { s[c] = d[c] != ValueType {} ? n[c] / d[c] : ValueType {1}; }
    );
}

/** @brief Fused update of the scale-corrected prolongation, per rank-local row i and column c:
 *    x = beta x + alpha (sf d + D^-1 (r - sf A d)),  sf per column, D^-1 per row.
 * alpha / beta are 1 x 1 or 1 x ncols. Replaces scale, copy, sub_scaled, scale, Diagonal apply,
 * scale and add_scaled (seven kernels). */
template<typename ValueType>
void fusedScaledUpdate(
    const gko::Executor& exec,
    localIdx rows,
    localIdx nc,
    ValueType* xv,
    localIdx xs,
    const ValueType* dv,
    localIdx ds,
    const ValueType* av,
    localIdx ads,
    const ValueType* rv,
    localIdx rs,
    const ValueType* sv,
    const ValueType* di,
    const ValueType* al,
    bool alphaPerCol,
    const ValueType* be,
    bool betaPerCol
)
{
    forEachOn(
        exec,
        static_cast<gko::size_type>(rows * nc),
        NEON_LAMBDA(const localIdx k) {
            const localIdx i = k / nc;
            const localIdx c = k % nc;
            const ValueType s = sv[c];
            const ValueType upd =
                s * dv[i * ds + c] + di[i] * (rv[i * rs + c] - s * av[i * ads + c]);
            const ValueType b = be[betaPerCol ? c : 0];
            const ValueType xo = b == ValueType {} ? ValueType {} : b * xv[i * xs + c];
            xv[i * xs + c] = xo + al[alphaPerCol ? c : 0] * upd;
        }
    );
}

/** @brief Set the square Dense @p m to the identity (on its executor). */
template<typename ValueType>
void setIdentity(gko::matrix::Dense<ValueType>* m)
{
    m->fill(ValueType {});
    auto* v = m->get_values();
    const localIdx stride = static_cast<localIdx>(m->get_stride());
    forEachOn(
        *m->get_executor(),
        m->get_size()[0],
        NEON_LAMBDA(const localIdx i) { v[i * stride + i] = ValueType {1}; }
    );
}

#ifdef NF_WITH_MPI_SUPPORT
/**
 * @brief Rank-local (block-diagonal) operator applied to distributed vectors WITHOUT communication.
 *
 * MergedPgm's distributed prolong/restrict have an empty off-diagonal block (aggregation is
 * rank-local), but as gko::experimental::distributed::Matrix every apply still runs the halo
 * machinery: an event sync, an MPI_Ialltoallv with zero counts (a collective over all ranks) and
 * MPI_Wait. This wrapper applies the local block to the local parts of the vectors instead: a plain
 * local SpMV.
 */
template<typename ValueType>
class LocalBlockOp : public gko::LinOp
{
    using dense = gko::matrix::Dense<ValueType>;
    using dist_vec = gko::experimental::distributed::Vector<ValueType>;

public:

    LocalBlockOp(std::shared_ptr<const gko::LinOp> localBlock, gko::dim<2> globalSize)
        : gko::LinOp(localBlock->get_executor(), globalSize), local_ {std::move(localBlock)}
    {}

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        local_->apply(
            gko::as<dist_vec>(b)->get_local_vector(), localView(gko::as<dist_vec>(x)).get()
        );
    }

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        local_->apply(
            alpha,
            gko::as<dist_vec>(b)->get_local_vector(),
            beta,
            localView(gko::as<dist_vec>(x)).get()
        );
    }

private:

    // mutable view of the rank-local part (distributed::Vector only exposes it read-only)
    static std::unique_ptr<dense> localView(dist_vec* v)
    {
        const auto* l = v->get_local_vector();
        return dense::create(
            l->get_executor(),
            l->get_size(),
            gko::make_array_view(
                l->get_executor(), l->get_size()[0] * l->get_stride(), v->get_local_values()
            ),
            l->get_stride()
        );
    }

    std::shared_ptr<const gko::LinOp> local_;
};
#endif

/** @brief Restriction that records the residual it restricts for ScaledProlongation. */
class RecordingRestriction : public gko::LinOp
{
public:

    RecordingRestriction(
        std::shared_ptr<const gko::LinOp> restrictOp, std::shared_ptr<ScaleCorrectionState> state
    )
        : gko::LinOp(restrictOp->get_executor(), restrictOp->get_size()),
          restrict_ {std::move(restrictOp)}, state_ {std::move(state)}
    {}

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        state_->residual = b;
        restrict_->apply(b, x);
    }

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        state_->residual = nullptr;
        restrict_->apply(alpha, b, beta, x);
    }

private:

    std::shared_ptr<const gko::LinOp> restrict_;
    std::shared_ptr<ScaleCorrectionState> state_;
};

/**
 * @brief Prolongation with OpenFOAM-style scale correction (GAMGSolverScale.C) of the coarse
 * correction.
 *
 * gko::solver::Multigrid adds the coarse correction through the advanced apply x += P e. When the
 * level's RecordingRestriction recorded the residual r it descended with, the prolonged correction
 * is Rayleigh-scaled and followed by one Jacobi step before it is added:
 *   d  = P e,  sf = (d . r) / (d . A d)   (per column; sf = 1 where d . A d == 0)
 *   x  = beta x + alpha (sf d + D^-1 (r - sf A d))
 * The simple apply is the plain prolongation (used e.g. by the level's composition).
 */
template<typename ValueType, typename IndexType>
class ScaledProlongation : public gko::LinOp
{
    using dense = gko::matrix::Dense<ValueType>;
    using diagonal = gko::matrix::Diagonal<ValueType>;

public:

    /** @param fineOp  the level's fine operator A (Csr, or distributed::Matrix)
     *  @param invDiag  D^-1 of A's (rank-local) diagonal block */
    ScaledProlongation(
        std::shared_ptr<const gko::LinOp> prolongOp,
        std::shared_ptr<const gko::LinOp> fineOp,
        std::shared_ptr<const diagonal> invDiag,
        std::shared_ptr<ScaleCorrectionState> state
    )
        : gko::LinOp(prolongOp->get_executor(), prolongOp->get_size()),
          prolong_ {std::move(prolongOp)}, fine_ {std::move(fineOp)}, invDiag_ {std::move(invDiag)},
          state_ {std::move(state)}, dotTmp_ {prolong_->get_executor()}
    {}

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override { prolong_->apply(b, x); }

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        const auto* r = state_->residual;
        state_->residual = nullptr;
        if (r == nullptr || r->get_size() != x->get_size())
        {
            prolong_->apply(alpha, b, beta, x);
            return;
        }
#ifdef NF_WITH_MPI_SUPPORT
        using dist_vec = gko::experimental::distributed::Vector<ValueType>;
        if (auto distX = dynamic_cast<dist_vec*>(x))
        {
            scaled(alpha, b, beta, distX, gko::as<const dist_vec>(r));
            return;
        }
#endif
        scaled(alpha, b, beta, gko::as<dense>(x), gko::as<const dense>(r));
    }

private:

    static dense* local(dense* v) { return v; }
    static const dense* local(const dense* v) { return v; }
#ifdef NF_WITH_MPI_SUPPORT
    static const dense* local(const gko::experimental::distributed::Vector<ValueType>* v)
    {
        return v->get_local_vector();
    }
    // mutable view of the rank-local part (distributed::Vector only exposes it read-only)
    static std::unique_ptr<dense> local(gko::experimental::distributed::Vector<ValueType>* v)
    {
        const auto* l = v->get_local_vector();
        return dense::create(
            l->get_executor(),
            l->get_size(),
            gko::make_array_view(
                l->get_executor(), l->get_size()[0] * l->get_stride(), v->get_local_values()
            ),
            l->get_stride()
        );
    }
#endif

    template<typename VecType>
    VecType* workspace(std::unique_ptr<gko::LinOp>& slot, const VecType* like) const
    {
        auto ws = dynamic_cast<VecType*>(slot.get());
        if (ws == nullptr || ws->get_size() != like->get_size())
        {
            slot = VecType::create_with_config_of(like);
            ws = gko::as<VecType>(slot.get());
        }
        return ws;
    }

    // Reduce the two local dot products [d.r | d.Ad] over all ranks in one allreduce (as Ginkgo's
    // all_reduce_sum, but one call for both instead of one per compute_dot).
    void reduce(const dense*, dense*) const {} // single-rank operator: the local dots are global
#ifdef NF_WITH_MPI_SUPPORT
    void reduce(const gko::experimental::distributed::Vector<ValueType>* x, dense* v) const
    {
        auto exec = v->get_executor();
        const auto comm = x->get_communicator();
        const int count = static_cast<int>(v->get_size()[1]);
        exec->synchronize();
        if (gko::experimental::mpi::requires_host_buffer(exec, comm))
        {
            if (!hostDots_ || hostDots_->get_size() != v->get_size())
            {
                hostDots_ = dense::create(exec->get_master(), v->get_size());
            }
            hostDots_->copy_from(v);
            comm.all_reduce(exec->get_master(), hostDots_->get_values(), count, MPI_SUM);
            v->copy_from(hostDots_);
        }
        else
        {
            comm.all_reduce(exec, v->get_values(), count, MPI_SUM);
        }
    }
#endif

    template<typename VecType>
    void scaled(
        const gko::LinOp* alpha,
        const gko::LinOp* e,
        const gko::LinOp* beta,
        VecType* x,
        const VecType* r
    ) const
    {
        auto exec = this->get_executor();
        const auto ncols = x->get_size()[1];
        auto* d = workspace(delta_, x);
        auto* ad = workspace(aDelta_, x);

        prolong_->apply(e, d); // d = P e
        fine_->apply(d, ad);   // ad = A d

        // both dot products into one 1 x 2 ncols buffer [d.r | d.Ad] and ONE reduction over the
        // ranks (two compute_dot calls cost two stream syncs + two allreduces); buffers are kept
        // between calls instead of being created (and `one` copied from the host) on every call
        if (!dots_ || dots_->get_size()[1] != 2 * ncols)
        {
            dots_ = dense::create(exec, gko::dim<2> {1, 2 * ncols});
            sf_ = dense::create(exec, gko::dim<2> {1, ncols});
        }
        auto num = dots_->create_submatrix(gko::span {0, 1}, gko::span {0, ncols});
        auto den = dots_->create_submatrix(gko::span {0, 1}, gko::span {ncols, 2 * ncols});
        const auto* dl = local(static_cast<const VecType*>(d));
        dl->compute_dot(local(r), num, dotTmp_);
        dl->compute_dot(local(static_cast<const VecType*>(ad)), den, dotTmp_);
        reduce(static_cast<const VecType*>(x), dots_.get());
        rayleighScale(num.get(), den.get(), sf_.get());

        // one fused kernel instead of scale/copy/sub_scaled/scale/Diagonal apply/scale/add_scaled:
        //   x = beta x + alpha (sf d + D^-1 (r - sf A d))   (rank-local rows, per column sf)
        auto alphaD = gko::make_temporary_clone(exec, gko::as<dense>(alpha));
        auto betaD = gko::make_temporary_clone(exec, gko::as<dense>(beta));
        auto&& xl = local(x);
        dense* xp = &*xl;
        const dense* rp = local(r);
        const dense* adp = local(static_cast<const VecType*>(ad));
        const localIdx nc = static_cast<localIdx>(ncols);
        const localIdx xs = static_cast<localIdx>(xp->get_stride());
        const localIdx ds = static_cast<localIdx>(dl->get_stride());
        const localIdx ads = static_cast<localIdx>(adp->get_stride());
        const localIdx rs = static_cast<localIdx>(rp->get_stride());
        const bool alphaPerCol = alphaD->get_size()[1] > 1;
        const bool betaPerCol = betaD->get_size()[1] > 1;
        detail::fusedScaledUpdate<ValueType>(
            *exec,
            static_cast<localIdx>(xp->get_size()[0]),
            nc,
            xp->get_values(),
            xs,
            dl->get_const_values(),
            ds,
            adp->get_const_values(),
            ads,
            rp->get_const_values(),
            rs,
            sf_->get_const_values(),
            invDiag_->get_const_values(),
            alphaD->get_const_values(),
            alphaPerCol,
            betaD->get_const_values(),
            betaPerCol
        );
    }

    std::shared_ptr<const gko::LinOp> prolong_;
    std::shared_ptr<const gko::LinOp> fine_;
    std::shared_ptr<const diagonal> invDiag_;
    std::shared_ptr<ScaleCorrectionState> state_;
    mutable std::unique_ptr<gko::LinOp> delta_;
    mutable std::unique_ptr<gko::LinOp> aDelta_;
    mutable std::unique_ptr<dense> dots_;     // [d.r | d.Ad], 1 x 2 ncols
    mutable std::unique_ptr<dense> sf_;       // Rayleigh factors, 1 x ncols
    mutable std::unique_ptr<dense> hostDots_; // host staging if MPI is not GPU-aware
    mutable gko::array<char> dotTmp_;         // compute_dot workspace
};

/** @brief D^-1 of a Csr's diagonal (zero where the diagonal is zero). */
template<typename ValueType, typename IndexType>
std::shared_ptr<const gko::matrix::Diagonal<ValueType>>
invertedDiagonal(const gko::matrix::Csr<ValueType, IndexType>* mtx)
{
    // invert in place on mtx's executor (no copy of the diagonal to the host and back)
    auto diag = gko::share(mtx->extract_diagonal());
    auto* v = diag->get_values();
    forEachOn(
        *diag->get_executor(),
        diag->get_size()[0],
        NEON_LAMBDA(const localIdx i) {
            v[i] = v[i] != ValueType {} ? ValueType {1} / v[i] : ValueType {};
        }
    );
    return diag;
}

} // namespace detail

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
 *
 * With scale_correction, the level's transfer operators apply OpenFOAM's GAMG scaleCorrection to
 * the coarse correction (see detail::ScaledProlongation). Unlike OpenFOAM, it also applies on the
 * level right above the coarsest (a level cannot tell its position in the hierarchy), and the
 * downward pre-smoother correction is not scaled (gko::solver::Multigrid never hands it to a
 * level).
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
        /** Scale the coarse correction like OpenFOAM's GAMG scaleCorrection. */
        bool GKO_FACTORY_PARAMETER_SCALAR(scale_correction, false);
        /** Distributed: apply the (block-diagonal) prolong/restrict as rank-local SpMVs without the
         * empty halo exchange of a distributed::Matrix (same result; default off). */
        bool GKO_FACTORY_PARAMETER_SCALAR(local_transfer, false);
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
        setLevel(A, A.get(), coarse);
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
        setLevel(system_matrix_, gko::as<csr>(distFine->get_diag_matrix().get()), coarse);
        recordReuse(reuseData, std::move(newSlots), true);
    }
#else
    void generateDistributed(reuse_data_type*) { GKO_NOT_IMPLEMENTED; }
#endif

private:

    // Set the level's operators; with scale_correction the transfer operators are wrapped (the
    // plain prolong_/restrict_ stay what the reuse data records). `diagBlock` is the rank-local
    // diagonal block of `fineOp` (the matrix itself in the local path).
    void setLevel(
        std::shared_ptr<const gko::LinOp> fineOp,
        const csr* diagBlock,
        std::shared_ptr<const gko::LinOp> coarse
    )
    {
        std::shared_ptr<const gko::LinOp> P = prolong_;
        std::shared_ptr<const gko::LinOp> R = restrict_;
#ifdef NF_WITH_MPI_SUPPORT
        if (parameters_.local_transfer)
        {
            auto dp = std::dynamic_pointer_cast<const dist_mtx>(prolong_);
            auto dr = std::dynamic_pointer_cast<const dist_mtx>(restrict_);
            auto emptyOffDiag = [](const dist_mtx* m)
            {
                const auto od = m->get_off_diag_matrix();
                if (od->get_size()[1] == 0 || od->get_size()[0] == 0) return true;
                auto odCsr = std::dynamic_pointer_cast<const csr>(od);
                return odCsr && odCsr->get_num_stored_elements() == 0;
            };
            if (dp && dr && emptyOffDiag(dp.get()) && emptyOffDiag(dr.get()))
            {
                P = std::make_shared<detail::LocalBlockOp<ValueType>>(
                    dp->get_diag_matrix(), dp->get_size()
                );
                R = std::make_shared<detail::LocalBlockOp<ValueType>>(
                    dr->get_diag_matrix(), dr->get_size()
                );
            }
            else
            {
                std::cerr << "NeoN MergedPgm: local_transfer requested but the transfer operators "
                             "are not "
                             "block-diagonal distributed matrices; keeping them\n";
            }
        }
#endif
        if (!parameters_.scale_correction)
        {
            this->set_multigrid_level(P, std::move(coarse), R);
            return;
        }
        auto state = std::make_shared<detail::ScaleCorrectionState>();
        this->set_multigrid_level(
            std::make_shared<detail::ScaledProlongation<ValueType, IndexType>>(
                P, std::move(fineOp), detail::invertedDiagonal(diagBlock), state
            ),
            std::move(coarse),
            std::make_shared<detail::RecordingRestriction>(R, state)
        );
    }

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
 * @brief Exact solver for a small (rank-local) matrix through its explicit dense inverse.
 *
 * generate() factorises the matrix with Ginkgo's sparse LU (solver::Direct, factorization::Lu) and
 * applies it to the identity (n right-hand sides), which gives A^-1 as a Dense n x n matrix;
 * apply() is then a single dense matrix-vector product (no iterations, no residual checks). Meant
 * as the local solver of the coarsest multigrid level, e.g. "coarsest_solver": {"type":
 * "preconditioner::Schwarz", "local_solver": {"type": "neon::DenseInverse"}} Memory n^2 values (2 k
 * rows: 32 MB in double).
 */
template<typename ValueType = gko::default_precision, typename IndexType = gko::int32>
class DenseInverse : public gko::LinOp
{
public:

    using value_type = ValueType;
    using index_type = IndexType;
    using dense = gko::matrix::Dense<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /** LU symbolic phase for a structurally symmetric matrix (otherwise the general algorithm).
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(symmetric, true);
    };
    GKO_ENABLE_LIN_OP_FACTORY(DenseInverse, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    std::shared_ptr<const dense> get_inverse() const { return inverse_; }

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override { inverse_->apply(b, x); }
    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        inverse_->apply(alpha, b, beta, x);
    }

    explicit DenseInverse(std::shared_ptr<const gko::Executor> exec) : gko::LinOp(std::move(exec))
    {}

    DenseInverse(const Factory* factory, std::shared_ptr<const gko::LinOp> system_matrix)
        : gko::LinOp(factory->get_executor(), system_matrix->get_size()),
          parameters_ {factory->get_parameters()}
    {
        GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
        auto exec = this->get_executor();
        const auto n = system_matrix->get_size()[0];
        using lu = gko::experimental::factorization::Lu<ValueType, IndexType>;
        auto direct = gko::experimental::solver::Direct<ValueType, IndexType>::build()
                          .with_factorization(lu::build().with_symbolic_algorithm(
                              parameters_.symmetric
                                  ? gko::experimental::factorization::symbolic_type::symmetric
                                  : gko::experimental::factorization::symbolic_type::general
                          ))
                          .with_num_rhs(n)
                          .on(exec)
                          ->generate(system_matrix);
        auto identity = dense::create(exec, gko::dim<2> {n, n});
        detail::setIdentity(identity.get());
        auto inverse = dense::create(exec, gko::dim<2> {n, n});
        direct->apply(identity, inverse);
        inverse_ = std::move(inverse);
    }

private:

    std::shared_ptr<const dense> inverse_;
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
    std::shared_ptr<const gko::Executor> exec,
    unsigned mergeLevels,
    bool deterministic = true,
    bool scaleCorrection = false
)
{
    return MergedPgm<ValueType, IndexType>::build()
        .with_merge_levels(mergeLevels)
        .with_deterministic(deterministic)
        .with_scale_correction(scaleCorrection)
        .on(std::move(exec));
}

/**
 * @brief Config-file support for `{"type": "neon::MergedPgm", ...}` and `{"type":
 * "neon::DenseInverse"}`, for the gko::config::registry constructor. Keys: merge_levels,
 * max_iterations, max_unassigned_ratio, deterministic, skip_sorting, scale_correction (see
 * MergedPgm::parameters).
 */
template<typename ValueType = gko::default_precision, typename IndexType = gko::int32>
inline gko::config::configuration_map mergedPgmConfigMap()
{
    return {
        {"neon::MergedPgm",
         [](const gko::config::pnode& config,
            const gko::config::registry&,
            gko::config::type_descriptor) -> gko::deferred_factory_parameter<gko::LinOpFactory>
         {
             auto params = MergedPgm<ValueType, IndexType>::build();
             auto toUnsigned = [](const gko::config::pnode& node, const std::string& key)
             {
                 const auto v = node.get_integer();
                 if (v < 0)
                 {
                     throw std::invalid_argument("neon::MergedPgm: " + key + " must be >= 0");
                 }
                 return static_cast<unsigned>(v);
             };
             for (const auto& [key, value] : config.get_map())
             {
                 if (key == "type" || key == "value_type") continue;
                 if (key == "merge_levels") params.with_merge_levels(toUnsigned(value, key));
                 else if (key == "max_iterations")
                     params.with_max_iterations(toUnsigned(value, key));
                 else if (key == "max_unassigned_ratio")
                     params.with_max_unassigned_ratio(value.get_real());
                 else if (key == "deterministic")
                     params.with_deterministic(value.get_boolean());
                 else if (key == "skip_sorting")
                     params.with_skip_sorting(value.get_boolean());
                 else if (key == "scale_correction")
                     params.with_scale_correction(value.get_boolean());
                 else if (key == "local_transfer")
                     params.with_local_transfer(value.get_boolean());
                 else
                     throw std::invalid_argument("neon::MergedPgm: unknown key '" + key + "'");
             }
             return params;
         }},
        {"neon::DenseInverse",
         [](const gko::config::pnode& config,
            const gko::config::registry&,
            gko::config::type_descriptor) -> gko::deferred_factory_parameter<gko::LinOpFactory>
         {
             auto params = DenseInverse<ValueType, IndexType>::build();
             for (const auto& [key, value] : config.get_map())
             {
                 if (key == "type" || key == "value_type") continue;
                 if (key == "symmetric") params.with_symmetric(value.get_boolean());
                 else
                     throw std::invalid_argument("neon::DenseInverse: unknown key '" + key + "'");
             }
             return params;
         }}
    };
}

} // namespace NeoN::la::ginkgo

#endif // NF_WITH_GINKGO
