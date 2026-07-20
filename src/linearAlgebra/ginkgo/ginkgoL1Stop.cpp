// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <stdexcept>

#include "NeoN/linearAlgebra/ginkgo.hpp"

namespace NeoN::la::ginkgo
{

#if NF_WITH_GINKGO

namespace
{

using dense = gko::matrix::Dense<scalar>;

#ifdef NF_WITH_MPI_SUPPORT
using dist = gko::experimental::distributed::Vector<scalar>;
#endif

// Small additive guard added to the norm factor so a trivially-converged (zero)
// system reports a finite, well-defined scaled residual instead of 0/0.
constexpr scalar normFactorSmall = 1e-20;

// Copies a [1 × ncols] Dense result to host and returns the sum of all columns.
// For scalar systems (ncols = 1) this is identical to retrieve(). For multi-column
// systems (e.g. Vec3 mapped as [nrows × 3]) it sums the per-column contributions to
// produce a single scalar norm/mean value for the stopping criterion.
scalar sumRetrieve(const dense* d)
{
    const auto ncols = d->get_size()[1];
    auto host = dense::create(d->get_executor()->get_master(), gko::dim<2> {1, ncols});
    host->copy_from(d);
    scalar total = 0;
    for (gko::size_type c = 0; c < ncols; ++c)
        total += host->at(0, c);
    return total;
}

/* @brief Create a vector shaped like @p like (same executor / distribution) with every
 * entry equal to @p value. Overloaded per concrete vector type so the criterion below
 * stays a single implementation shared by the serial (Dense) and distributed paths.
 */
std::unique_ptr<dense> makeConstantLike(const dense& like, scalar value)
{
    auto field = dense::create(like.get_executor(), like.get_size());
    field->fill(value);
    return field;
}

#ifdef NF_WITH_MPI_SUPPORT
std::unique_ptr<dist> makeConstantLike(const dist& like, scalar value)
{
    auto field = dist::create(
        like.get_executor(),
        like.get_communicator(),
        like.get_size(),
        like.get_local_vector()->get_size()
    );
    field->fill(value);
    return field;
}
#endif

/* @brief L1 normalisation factor of a linear system, evaluated at x.
 *
 *   normFactor_i = sum_rows( |(A x)_{row,i} - (A xRef)_{row,i}| + |b_{row,i} - (A xRef)_{row,i}| )
 *                 + small
 *
 * with xRef the constant field equal to mean(x) across all columns.  (A xRef) is the SpMV of A
 * with that constant field.  The factor measures the spread of A x and b about the reference state
 * A xRef and scales the L1 residual independently of the overall magnitude of the system.
 *
 * Returns:
 *   .first  — combined normFactor: sum of per-column factors (used as the single stopping scalar)
 *   .second — per-column normFactors (one entry per RHS column; size == ncols)
 *
 * Identical for serial and distributed systems: when VecType is a distributed vector,
 * compute_mean() and compute_norm1() perform the global (cross-rank) reductions, so the
 * factor is the global one with no extra MPI code here.
 */
template<typename VecType>
std::pair<scalar, std::vector<scalar>> computeL1NormFactor(
    std::shared_ptr<const gko::Executor> exec,
    const gko::LinOp* mtx,
    const VecType* b,
    const VecType* x,
    const VecType* r
)
{
    const auto one = gko::initialize<dense>({1.0}, exec);
    const auto ncols = x->get_size()[1];

    // xRef per column: for multi-RHS (e.g. Vec3 as [nrows×3]) each column gets its own
    // reference state equal to that column's mean. This matches OpenFOAM's per-component
    // L1 norm factor and ensures per-column residuals are computed correctly.
    // For single-column solves (ncols=1) this is identical to the scalar mean.
    auto meanDense = dense::create(exec, gko::dim<2> {1, ncols});
    x->compute_mean(meanDense);

    // xRefField[i][c] = mean(x[:,c]) for all rows i  (per-column constant field)
    auto xRefField = makeConstantLike(*x, 1.0);
    xRefField->scale(meanDense);
    auto Axref = makeConstantLike(*b, 0.0);
    mtx->apply(xRefField, Axref);

    // |b - A xRef|
    auto bMinusAxref = b->clone();
    bMinusAxref->sub_scaled(one, Axref);
    auto term2 = bMinusAxref->compute_absolute();

    // |A x - A xRef|, reusing the residual r = b - A x to avoid a second SpMV:
    //   A x - A xRef = (b - A xRef) - r
    bMinusAxref->sub_scaled(one, r);
    auto term = bMinusAxref->compute_absolute();
    term->add_scaled(one, term2);

    // Per-column norm1 (rows summed, one value per column; global sum for distributed)
    auto nf = dense::create(exec, gko::dim<2> {1, ncols});
    term->compute_norm1(nf);

    // Copy to host to extract per-column values
    auto nh = dense::create(exec->get_master(), gko::dim<2> {1, ncols});
    nh->copy_from(nf);

    std::vector<scalar> perCol(ncols);
    scalar combined = 0;
    for (gko::size_type c = 0; c < ncols; ++c)
    {
        perCol[c] = nh->at(0, c) + normFactorSmall;
        combined += perCol[c];
    }
    return {combined, perCol};
}

/* @brief Ginkgo stopping criterion based on the L1-scaled residual, shared by the serial
 * (VecType = gko::matrix::Dense) and distributed (VecType = distributed::Vector) solves.
 *
 * The iteration stops on the scaled residual sum|b - A x| / normFactor, using an absolute
 * tolerance, a relative tolerance (relative to the initial residual) and a maximum
 * iteration count, while honouring a minimum iteration count. The L1 residual norm is taken
 * from the solver's recurrent residual b - A x (Cg/BiCGStab keep it unpreconditioned, which
 * also matches OpenFOAM's recurrently-updated residual); it is recomputed with a single SpMV
 * only when the solver does not expose one (e.g. multigrid). For a distributed vector the
 * norms are global.
 */
template<typename VecType>
class L1ResidualCriterion :
    public gko::EnablePolymorphicObject<L1ResidualCriterion<VecType>, gko::stop::Criterion>
{
    friend class gko::EnablePolymorphicObject<L1ResidualCriterion<VecType>, gko::stop::Criterion>;
    using Criterion = gko::stop::Criterion;

public:

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        // NOTE: GKO_FACTORY_PARAMETER_SCALAR (single-arg setter) is required here;
        // the variadic GKO_FACTORY_PARAMETER expands its with_*() setter to
        // GKO_NOT_IMPLEMENTED under NVCC/HIP (parameter-pack workaround).
        scalar GKO_FACTORY_PARAMETER_SCALAR(absolute_tolerance, 1.0e-6);

        scalar GKO_FACTORY_PARAMETER_SCALAR(relative_tolerance, 0.0);

        gko::size_type GKO_FACTORY_PARAMETER_SCALAR(min_iter, 0);

        gko::size_type GKO_FACTORY_PARAMETER_SCALAR(max_iter, 1000);

        localIdx GKO_FACTORY_PARAMETER_SCALAR(check_frequency, 1);

        std::shared_ptr<const gko::LinOp> GKO_FACTORY_PARAMETER_SCALAR(matrix, nullptr);

        std::shared_ptr<const VecType> GKO_FACTORY_PARAMETER_SCALAR(b, nullptr);

        std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(init_residual, NULL);

        std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(final_residual, NULL);

        std::add_pointer<gko::size_type>::type GKO_FACTORY_PARAMETER_SCALAR(num_iters, NULL);

        // Per-column scaled residual output for multi-RHS (Vec3) solves.
        // When non-null, filled with one entry per column (size == ncols).
        std::add_pointer<std::vector<scalar>>::type GKO_FACTORY_PARAMETER_SCALAR(
            per_col_init_residuals, NULL
        );

        std::add_pointer<std::vector<scalar>>::type GKO_FACTORY_PARAMETER_SCALAR(
            per_col_final_residuals, NULL
        );
    };

    GKO_ENABLE_CRITERION_FACTORY(L1ResidualCriterion, parameters, Factory);

    GKO_ENABLE_BUILD_METHOD(Factory);

    explicit L1ResidualCriterion(std::shared_ptr<const gko::Executor> exec)
        : gko::EnablePolymorphicObject<L1ResidualCriterion<VecType>, Criterion>(std::move(exec))
    {}

    explicit L1ResidualCriterion(const Factory* factory, const gko::stop::CriterionArgs&)
        : gko::EnablePolymorphicObject<L1ResidualCriterion<VecType>, Criterion>(
            factory->get_executor()
        ),
          parameters_ {factory->get_parameters()}
    {}

protected:

    bool check_impl(
        gko::uint8 stoppingId,
        bool setFinalized,
        gko::array<gko::stopping_status>* stop_status,
        bool* one_changed,
        const Criterion::Updater& updater
    ) override
    {
        const auto exec = this->get_executor();

        // The current solution is needed for the normFactor reference state (and for the
        // residual recompute fallback). For the iterative solvers used here (Cg/BiCGStab)
        // it is set on every check that can stop.
        if (updater.solution_ == nullptr)
        {
            return false;
        }
        const auto* solution = gko::as<VecType>(updater.solution_);
        const auto numIter = updater.num_iterations_;

        // Skip the residual-norm evaluation on iterations where the criterion cannot stop
        // anyway: before min_iter the tolerances are not tested, and between check_frequency-
        // spaced checks. The very first call (to capture the initial residual and normFactor)
        // and the max_iter cap must always be evaluated.
        const bool mustEvaluate = firstIter_ || numIter >= parameters_.max_iter
                               || (numIter >= parameters_.min_iter
                                   && (parameters_.check_frequency <= 1
                                       || numIter % parameters_.check_frequency == 0));
        if (!mustEvaluate)
        {
            return false;
        }

        // Prefer the solver's recurrent residual r = b - A x. Ginkgo's Cg/BiCGStab keep it
        // up to date and unpreconditioned (they pass .residual(r)), so it is exactly the L1
        // numerator and matches OpenFOAM's recurrently-updated residual. Recomputing it with a
        // full SpMV on every check roughly doubled the per-iteration cost of the cheaply-
        // preconditioned solve. Fall back to the SpMV recompute only when the solver does not
        // expose a residual (e.g. multigrid), which then needs the current solution.
        std::unique_ptr<VecType> rOwned;
        const VecType* r = nullptr;
        if (updater.residual_ != nullptr)
        {
            r = gko::as<VecType>(updater.residual_);
        }
        else
        {
            const auto one = gko::initialize<dense>({1.0}, exec);
            const auto negOne = gko::initialize<dense>({-1.0}, exec);
            rOwned = parameters_.b->clone();
            parameters_.matrix->apply(negOne.get(), solution, one.get(), rOwned.get());
            r = rOwned.get();
        }

        const auto ncols = r->get_size()[1];
        auto rNormDense = dense::create(exec, gko::dim<2> {1, ncols});
        r->compute_norm1(rNormDense.get());
        const scalar rNorm = sumRetrieve(rNormDense.get());

        const bool isFirstIter = firstIter_;
        if (firstIter_)
        {
            auto [combined, perCol] = computeL1NormFactor<VecType>(
                exec, parameters_.matrix.get(), parameters_.b.get(), solution, r
            );
            normFactor_ = combined;
            perColNormFactor_ = std::move(perCol);
            initResidual_ = rNorm / normFactor_;
            if (parameters_.init_residual != NULL)
            {
                *(parameters_.init_residual) = initResidual_;
            }
            firstIter_ = false;
        }

        const scalar scaledResidual = rNorm / normFactor_;
        if (parameters_.final_residual != NULL)
        {
            *(parameters_.final_residual) = scaledResidual;
        }
        if (parameters_.num_iters != NULL)
        {
            *(parameters_.num_iters) = numIter;
        }

        // Per-column residuals for multi-RHS (Vec3) solves: copy rNormDense to host once and
        // divide each column by its own normFactor for accurate per-component reporting.
        const bool hasPerColOut = ncols > 1 && !perColNormFactor_.empty()
                               && (parameters_.per_col_init_residuals != NULL
                                   || parameters_.per_col_final_residuals != NULL);
        if (hasPerColOut)
        {
            auto nh = dense::create(exec->get_master(), gko::dim<2> {1, ncols});
            nh->copy_from(rNormDense);
            if (isFirstIter && parameters_.per_col_init_residuals != NULL)
            {
                auto& out = *parameters_.per_col_init_residuals;
                out.resize(ncols);
                for (gko::size_type c = 0; c < ncols; ++c)
                    out[c] = nh->at(0, c) / perColNormFactor_[c];
            }
            if (parameters_.per_col_final_residuals != NULL)
            {
                auto& out = *parameters_.per_col_final_residuals;
                out.resize(ncols);
                for (gko::size_type c = 0; c < ncols; ++c)
                    out[c] = nh->at(0, c) / perColNormFactor_[c];
            }
        }

        bool result = false;
        // stop if maximum number of iterations was reached
        if (numIter >= parameters_.max_iter)
        {
            result = true;
        }
        // only test the tolerances once the minimum iteration count is reached
        else if (numIter >= parameters_.min_iter)
        {
            if (scaledResidual < parameters_.absolute_tolerance)
            {
                result = true;
            }
            if (parameters_.relative_tolerance > 0.0
                && scaledResidual < parameters_.relative_tolerance * initResidual_)
            {
                result = true;
            }
        }

        if (result)
        {
            this->set_all_statuses(stoppingId, setFinalized, stop_status);
            *one_changed = true;
        }
        return result;
    }

private:

    mutable bool firstIter_ = true;

    mutable scalar normFactor_ = 1.0;

    mutable scalar initResidual_ = 0.0;

    mutable std::vector<scalar> perColNormFactor_;
};

/* @brief Attach an L1ResidualCriterion to @p solver, run it, and report the scaled L1
 * initial/final residual and iteration count. Shared by the serial and distributed entry
 * points below. @p x is updated in place.
 */
template<typename VecType>
L1ResidualResult attachL1StopAndSolve(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> mtx,
    std::shared_ptr<const VecType> b,
    std::shared_ptr<VecType> x,
    gko::LinOp* solver,
    const L1ResidualControl& control
)
{
    scalar initResNorm = 0.0;
    scalar finalResNorm = 0.0;
    gko::size_type numIter = 0;
    std::vector<scalar> perColInit, perColFinal;

    const bool multiRhs = b->get_size()[1] > 1;

    auto params = L1ResidualCriterion<VecType>::build()
                      .with_absolute_tolerance(control.tolerance)
                      .with_relative_tolerance(control.relTol)
                      .with_min_iter(control.minIter)
                      .with_max_iter(control.maxIter)
                      .with_check_frequency(control.checkFrequency)
                      .with_matrix(mtx)
                      .with_b(b)
                      .with_init_residual(&initResNorm)
                      .with_final_residual(&finalResNorm)
                      .with_num_iters(&numIter);

    if (multiRhs)
    {
        params.with_per_col_init_residuals(&perColInit).with_per_col_final_residuals(&perColFinal);
    }

    auto criterion = params.on(exec);

    auto* iterative = dynamic_cast<gko::solver::IterativeBase*>(solver);
    if (iterative == nullptr)
    {
        throw std::runtime_error("L1 scaled-residual stopping requires an iterative Ginkgo solver");
    }
    iterative->set_stop_criterion_factory(gko::share(std::move(criterion)));

    solver->apply(b, x);

    L1ResidualResult result {numIter, initResNorm, finalResNorm};
    if (multiRhs)
    {
        result.perColInitNorms = std::move(perColInit);
        result.perColFinalNorms = std::move(perColFinal);
    }
    return result;
}

} // namespace

L1ResidualResult solveWithL1Stop(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> mtx,
    std::shared_ptr<const gko::matrix::Dense<scalar>> b,
    std::shared_ptr<gko::matrix::Dense<scalar>> x,
    gko::LinOp* solver,
    const L1ResidualControl& control
)
{
    return attachL1StopAndSolve<dense>(exec, mtx, b, x, solver, control);
}

#ifdef NF_WITH_MPI_SUPPORT
L1ResidualResult solveWithL1StopDist(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> mtx,
    std::shared_ptr<const gko::experimental::distributed::Vector<scalar>> b,
    std::shared_ptr<gko::experimental::distributed::Vector<scalar>> x,
    gko::LinOp* solver,
    const L1ResidualControl& control
)
{
    return attachL1StopAndSolve<dist>(exec, mtx, b, x, solver, control);
}
#endif

#endif

} // namespace NeoN::la::ginkgo
