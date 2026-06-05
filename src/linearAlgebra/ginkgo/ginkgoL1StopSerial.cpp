// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <stdexcept>

#include "NeoN/linearAlgebra/ginkgo.hpp"

namespace NeoN::la::ginkgo
{

#if NF_WITH_GINKGO

namespace
{

using vec = gko::matrix::Dense<scalar>;

// Small additive guard added to the norm factor so a trivially-converged (zero)
// system reports a finite, well-defined scaled residual instead of 0/0.
constexpr scalar normFactorSmall = 1e-20;

/* @brief L1 normalisation factor of a (rank-local) serial system, evaluated at x.
 *
 *   normFactor = sum_i ( |(A x)_i - (A xRef)_i| + |b_i - (A xRef)_i| ) + small
 *
 * with xRef the constant field equal to mean(x); (A xRef) is the SpMV of A with that
 * constant field, which equals sumA_i * mean(x). The factor measures the spread of
 * A x and b about the reference state A xRef and is used to scale the L1 residual so
 * the reported value is independent of the overall magnitude of the system. This is
 * the Dense-vector analogue of compute_normfactor_dist() in ginkgoL1Stop.cpp.
 */
scalar computeL1NormFactor(
    std::shared_ptr<const gko::Executor> exec, const gko::LinOp* mtx, const vec* b, const vec* x
)
{
    const auto one = gko::initialize<vec>({1.0}, exec);

    // xRef = mean(x)
    auto meanDense = vec::create(exec, gko::dim<2> {1});
    x->compute_mean(meanDense);
    const scalar xRef = retrieve(meanDense.get());

    // A xRef, with xRef broadcast to a constant field
    auto xRefVec = vec::create(exec, x->get_size());
    xRefVec->fill(xRef);
    auto Axref = vec::create(exec, b->get_size());
    mtx->apply(xRefVec, Axref);

    // |A x - A xRef|
    auto Apsi = vec::create(exec, b->get_size());
    mtx->apply(x, Apsi);
    Apsi->sub_scaled(one, Axref);
    auto term = Apsi->compute_absolute();

    // + |b - A xRef|
    auto bMinusAxref = b->clone();
    bMinusAxref->sub_scaled(one, Axref);
    auto term2 = bMinusAxref->compute_absolute();
    term->add_scaled(one, term2);

    // sum over rows (entries already non-negative)
    auto nf = vec::create(exec, gko::dim<2> {1});
    term->compute_norm1(nf);
    return retrieve(nf.get()) + normFactorSmall;
}

/* @brief Ginkgo stopping criterion based on the L1-scaled residual.
 *
 * The iteration is stopped on the scaled residual sum|b - A x| / normFactor, using an
 * absolute tolerance, a relative tolerance (relative to the initial residual) and a
 * maximum iteration count, while honouring a minimum iteration count. The true residual
 * b - A x is recomputed every check so the criterion is independent of any preconditioned
 * residual the solver may carry internally.
 */
class L1ResidualCriterion :
    public gko::EnablePolymorphicObject<L1ResidualCriterion, gko::stop::Criterion>
{
    friend class gko::EnablePolymorphicObject<L1ResidualCriterion, gko::stop::Criterion>;
    using Criterion = gko::stop::Criterion;

public:

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        // NOTE: GKO_FACTORY_PARAMETER_SCALAR (single-arg setter) is required here;
        // the variadic GKO_FACTORY_PARAMETER expands its with_*() setter to
        // GKO_NOT_IMPLEMENTED under NVCC/HIP (parameter-pack workaround).
        scalar GKO_FACTORY_PARAMETER_SCALAR(absolute_tolerance, 1.0e-6);

        scalar GKO_FACTORY_PARAMETER_SCALAR(relative_tolerance, 0.0);

        localIdx GKO_FACTORY_PARAMETER_SCALAR(min_iter, 0);

        localIdx GKO_FACTORY_PARAMETER_SCALAR(max_iter, 1000);

        std::shared_ptr<const gko::LinOp> GKO_FACTORY_PARAMETER_SCALAR(matrix, nullptr);

        std::shared_ptr<const vec> GKO_FACTORY_PARAMETER_SCALAR(b, nullptr);

        std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(init_residual, NULL);

        std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(final_residual, NULL);

        std::add_pointer<localIdx>::type GKO_FACTORY_PARAMETER_SCALAR(num_iters, NULL);
    };

    GKO_ENABLE_CRITERION_FACTORY(L1ResidualCriterion, parameters, Factory);

    GKO_ENABLE_BUILD_METHOD(Factory);

    explicit L1ResidualCriterion(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<L1ResidualCriterion, Criterion>(std::move(exec))
    {}

    explicit L1ResidualCriterion(const Factory* factory, const gko::stop::CriterionArgs&)
        : EnablePolymorphicObject<L1ResidualCriterion, Criterion>(factory->get_executor()),
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

        // We need the current solution to evaluate the true residual b - A x.
        // For the iterative solvers used here (Cg/BiCGStab/Gmres) it is always set.
        if (updater.solution_ == nullptr)
        {
            return false;
        }
        const auto* solution = gko::as<vec>(updater.solution_);
        const auto numIter = static_cast<localIdx>(updater.num_iterations_);

        // true residual r = b - A x (recomputed independently of the solver's residual)
        const auto one = gko::initialize<vec>({1.0}, exec);
        const auto negOne = gko::initialize<vec>({-1.0}, exec);
        auto r = parameters_.b->clone();
        parameters_.matrix->apply(negOne.get(), solution, one.get(), r.get());

        auto rNormDense = vec::create(exec, gko::dim<2> {1});
        r->compute_norm1(rNormDense.get());
        const scalar rNorm = retrieve(rNormDense.get());

        if (firstIter_)
        {
            normFactor_ =
                computeL1NormFactor(exec, parameters_.matrix.get(), parameters_.b.get(), solution);
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
};

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
    scalar initResNorm = 0.0;
    scalar finalResNorm = 0.0;
    localIdx numIter = 0;

    auto criterion = L1ResidualCriterion::build()
                         .with_absolute_tolerance(control.tolerance)
                         .with_relative_tolerance(control.relTol)
                         .with_min_iter(control.minIter)
                         .with_max_iter(control.maxIter)
                         .with_matrix(mtx)
                         .with_b(b)
                         .with_init_residual(&initResNorm)
                         .with_final_residual(&finalResNorm)
                         .with_num_iters(&numIter)
                         .on(exec);

    auto* iterative = dynamic_cast<gko::solver::IterativeBase*>(solver);
    if (iterative == nullptr)
    {
        throw std::runtime_error("L1 scaled-residual stopping requires an iterative Ginkgo solver");
    }
    iterative->set_stop_criterion_factory(gko::share(std::move(criterion)));

    solver->apply(b, x);

    return {numIter, initResNorm, finalResNorm};
}

#endif

} // namespace NeoN::la::ginkgo
