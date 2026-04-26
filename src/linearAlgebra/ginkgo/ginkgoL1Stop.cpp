// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT


#include "NeoN/linearAlgebra/ginkgo.hpp"

namespace NeoN::la::ginkgo
{

#if NF_WITH_GINKGO

class StoppingCriterion
{
    using vec = gko::matrix::Dense<scalar>;
    using mtx = gko::matrix::Csr<scalar>;
    using val_array = gko::array<scalar>;
    using idx_array = gko::array<localIdx>;

    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    class DistStoppingCriterion :
        public gko::EnablePolymorphicObject<DistStoppingCriterion, gko::stop::Criterion>
    {
        friend class gko::EnablePolymorphicObject<DistStoppingCriterion, gko::stop::Criterion>;
        using Criterion = gko::stop::Criterion;

    public:

        GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
        {
            /**
             * Boolean set by the user to stop the iteration process
             */
            // TODO check why GKO_FACTORY_PARAMETER_SCALAR does not work
            scalar GKO_FACTORY_PARAMETER(absolute_tolerance, 1.0e-6);

            scalar GKO_FACTORY_PARAMETER(relative_tolerance, 0.0);

            localIdx GKO_FACTORY_PARAMETER(minIter, 0);

            localIdx GKO_FACTORY_PARAMETER(maxIter, 0);

            localIdx GKO_FACTORY_PARAMETER(frequency, 1);

            std::add_pointer<localIdx>::type GKO_FACTORY_PARAMETER_SCALAR(iter, NULL);

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(time, NULL);

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(residual_norm, NULL);

            std::shared_ptr<vec> GKO_FACTORY_PARAMETER_SCALAR(residual_norms, {});

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(init_residual_norm, NULL);

            localIdx GKO_FACTORY_PARAMETER(verbose, 0);

            bool GKO_FACTORY_PARAMETER(export_res, false);

            std::shared_ptr<const gko::LinOp> GKO_FACTORY_PARAMETER(gkomatrix, {});

            std::shared_ptr<dist_vec> GKO_FACTORY_PARAMETER(x, {});

            std::shared_ptr<dist_vec> GKO_FACTORY_PARAMETER(b, {});
        };

        GKO_ENABLE_CRITERION_FACTORY(DistStoppingCriterion, parameters, Factory);

        GKO_ENABLE_BUILD_METHOD(Factory);

        /* Compute the SpMV of A with x_ref, where x_ref is a vector containing
         * the average of x in every row. This is needed to initialise the
         * normfactor in the first iteration.
         *  */
        void compute_Axref_dist(
            size_t global_size,
            size_t local_size,
            std::shared_ptr<const gko::Executor> device_exec,
            std::shared_ptr<const gko::LinOp> gkomatrix,
            std::shared_ptr<const dist_vec> x,
            std::shared_ptr<dist_vec> res
        ) const;

        /* Compute the normfactor ie || Ax - x* || + || b - x* ||
         * or rewritten as || r - ( b - x* ) || + || (b - x*) ||
         *  */
        scalar compute_normfactor_dist(
            std::shared_ptr<const gko::Executor> device_exec,
            const dist_vec* r,
            std::shared_ptr<const gko::LinOp> gkomatrix,
            std::shared_ptr<const dist_vec> x,
            std::shared_ptr<const dist_vec> b
        ) const;

        /* Implementation of the residual norm check
         *  */
        bool check_impl(
            gko::uint8 stoppingId,
            bool setFinalized,
            gko::array<gko::stopping_status>* stop_status,
            bool* one_changed,
            const Criterion::Updater& updater
        ) override;


        explicit DistStoppingCriterion(std::shared_ptr<const gko::Executor> exec)
            : EnablePolymorphicObject<DistStoppingCriterion, Criterion>(std::move(exec))
        {}

        explicit DistStoppingCriterion(const Factory* factory, const gko::stop::CriterionArgs&)

            : EnablePolymorphicObject<DistStoppingCriterion, Criterion>(factory->get_executor()),
              parameters_ {factory->get_parameters()}
        {}

        void set_eval_norm_factor(bool eval_norm_factor) { eval_norm_factor_ = eval_norm_factor; }

        mutable bool first_iter_ = true;

        mutable scalar norm_factor_ = 1;

        mutable bool eval_norm_factor_ = true;

        mutable std::vector<scalar> res_norms_ {};
    };

    mutable localIdx maxIter_;

    const localIdx minIter_;

    const scalar tolerance_;

    const scalar relTol_;

    const scalar res_norm_eval_;

    const localIdx norm_eval_limit_;

    const localIdx frequency_;

    const scalar relaxationFactor_;

    const bool adapt_minIter_;

    const std::shared_ptr<vec> normalised_res_norms_;

    mutable scalar init_normalised_res_norm_;

    mutable scalar normalised_res_norm_;

    mutable localIdx iter_;

    mutable scalar time_;

public:

    StoppingCriterion(const Dictionary& controlDict)
        : maxIter_(controlDict.get<localIdx>("maxIter", 1000)),
          minIter_(controlDict.get<localIdx>("minIter", 0)),
          tolerance_(controlDict.get<scalar>("tolerance", 1e-6)),
          relTol_(controlDict.get<scalar>("relTol", 1e-6)),
          res_norm_eval_(controlDict.get<scalar>("resNormEval", 0.1)),
          norm_eval_limit_(controlDict.get<localIdx>("normEvalLimit", 100)),
          frequency_(controlDict.get<localIdx>("evalFrequency", 1)),
          relaxationFactor_(controlDict.get<scalar>("relaxationFactor", 0.6)),
          adapt_minIter_(controlDict.get<bool>("adaptMinIter", true)),
          normalised_res_norms_(gko::share(vec::create(
              gko::ReferenceExecutor::create(),
              gko::dim<2> {static_cast<gko::dim<2>::dimension_type>(maxIter_), 1}
          ))),
          init_normalised_res_norm_(0), normalised_res_norm_(0), iter_(0), time_(0)
    {
        normalised_res_norms_->fill(0.0);
        if (controlDict.get<std::string>("solver") == "GKOBiCGStab") maxIter_ *= 2;
    }

    std::shared_ptr<const gko::stop::CriterionFactory> build_dist_stopping_criterion(
        std::shared_ptr<gko::Executor> device_exec,
        std::shared_ptr<const gko::LinOp> gkomatrix,
        std::shared_ptr<dist_vec> x,
        std::shared_ptr<dist_vec> b,
        label verbose,
        bool export_res,
        label prev_solve_iters,
        scalar prev_rel_cost
    ) const
    {
        std::string frequencyMode = "optimizer";
        label minIter = minIter_;
        label frequency = frequency_;
        // in case of export_res all residuals need to be computed
        if (!export_res)
        {
            if (prev_solve_iters > 0 && adapt_minIter_ && prev_rel_cost > 0)
            {
                minIter = prev_solve_iters * relaxationFactor_;
                if (frequencyMode == "optimizer")
                {
                    auto alpha =
                        sqrt(1.0 / (prev_solve_iters * (1.0 - relaxationFactor_)) * prev_rel_cost);
                    frequency = std::min(norm_eval_limit_, std::max(1, localIdx(1 / alpha)));
                }
                if (frequencyMode == "relative")
                {
                    frequency = localIdx(prev_solve_iters * 0.075) + 1;
                }
            }
        }

        std::string msg = "\nCreating stopping criterion\n\tminIter: " + std::to_string(minIter)
                        + "\n\tfrequency: " + std::to_string(frequency) + "\n\tprev_solve_iters: "
                        + std::to_string(prev_solve_iters) + "\n\tadapt_minIter:  "
                        + std::to_string(adapt_minIter_) + "\n\tprev_rel_cost: ";

        NeoN::Logging::info(msg);

        return DistStoppingCriterion::build()
            .with_absolute_tolerance(tolerance_)
            .with_relative_tolerance(relTol_)
            .with_minIter(minIter)
            .with_maxIter(maxIter_)
            .with_frequency(frequency)
            .with_verbose(verbose)
            .with_export_res(export_res)
            .with_init_residual_norm(&init_normalised_res_norm_)
            .with_residual_norm(&normalised_res_norm_)
            .with_residual_norms(normalised_res_norms_)
            .with_iter(&iter_)
            .with_time(&time_)
            .with_gkomatrix(gkomatrix)
            .with_x(x)
            .with_b(b)
            .on(device_exec);
    }

    scalar get_init_res_norm() const { return init_normalised_res_norm_; }

    scalar get_res_norm() const { return normalised_res_norm_; }

    std::shared_ptr<vec> get_res_norms() const { return normalised_res_norms_; }

    label get_is_final() const { return relTol_ == 0.0; }

    label get_num_iters() const { return iter_; }

    scalar get_res_norm_time() const { return time_; }
};


void StoppingCriterion::DistStoppingCriterion::compute_Axref_dist(
    size_t global_size,
    size_t local_size,
    std::shared_ptr<const gko::Executor> device_exec,
    std::shared_ptr<const gko::LinOp> gkomatrix,
    std::shared_ptr<const dist_vec> x,
    std::shared_ptr<dist_vec> res
) const
{
    auto xAvg = gko::initialize<gko::matrix::Dense<scalar>>(1, {0}, device_exec);
    x->compute_mean(xAvg.get());

    auto xAvg_host = gko::initialize<gko::matrix::Dense<scalar>>(1, {0}, device_exec->get_master());
    xAvg->move_to(xAvg_host);
    auto xAvg_vec = gko::share(dist_vec::create(
        device_exec,
        x->get_communicator(),
        gko::dim<2> {global_size, 1},
        gko::dim<2> {local_size, 1}
    ));
    xAvg_vec->fill(xAvg_host->at(0));

    gkomatrix->apply(xAvg_vec.get(), res.get());
}

scalar StoppingCriterion::DistStoppingCriterion::compute_normfactor_dist(
    std::shared_ptr<const gko::Executor> device_exec,
    const dist_vec* r,
    std::shared_ptr<const gko::LinOp> gkomatrix,
    std::shared_ptr<const dist_vec> x,
    std::shared_ptr<const dist_vec> b
) const
{
    // TODO store colA vector
    auto comm = x->get_communicator();

    gko::dim<2> local_size = x->get_local_vector()->get_size();
    gko::dim<2> global_size = x->get_size();

    auto Axref = gko::share(dist_vec::create(device_exec, comm, global_size, local_size));
    Axref->fill(0.0);

    auto start_axref = std::chrono::steady_clock::now();
    compute_Axref_dist(global_size[0], local_size[0], device_exec, gkomatrix, x, Axref);
    auto end_axref = std::chrono::steady_clock::now();
    auto delta_t_axref =
        std::chrono::duration_cast<std::chrono::microseconds>(end_axref - start_axref).count()
        / 1.0;
    // std::cout << __FILE__ << " delta_t_axref " << delta_t_axref << " [mu
    // s]\n";

    auto unity = gko::initialize<gko::matrix::Dense<scalar>>(1, {1.0}, device_exec);

    auto b_sub_xstar = b->clone();
    b_sub_xstar->sub_scaled(unity.get(), Axref.get());

    auto norm_part2 = b_sub_xstar->compute_absolute();

    b_sub_xstar->sub_scaled(unity.get(), r);

    b_sub_xstar->compute_absolute_inplace();
    b_sub_xstar->add_scaled(unity.get(), norm_part2.get());

    auto res = vec::create(device_exec, gko::dim<2> {1});
    b_sub_xstar->compute_norm1(res.get());

    auto res_host = vec::create(device_exec->get_master(), gko::dim<2> {1});
    res_host->copy_from(res.get());

    return res_host->get_values()[0] + ROOTVSMALL;
}

bool StoppingCriterion::DistStoppingCriterion::check_impl(
    gko::uint8 stoppingId,
    bool setFinalized,
    gko::array<gko::stopping_status>* stop_status,
    bool* one_changed,
    const Criterion::Updater& updater
)
{
    // Dont check residual norm before minIter is reached
    if (*(parameters_.iter) > 0 && *(parameters_.iter) < parameters_.minIter)
    {
        *(parameters_.iter) += 1;
        return false;
    }

    // Only check residual for every frequency iteration
    if (*(parameters_.iter) % parameters_.frequency != 0)
    {
        *(parameters_.iter) += 1;
        return false;
    }

    auto start_eval = std::chrono::steady_clock::now();
    const auto exec = this->get_executor();

    std::shared_ptr<dist_vec> dense_r_vec;
    // multigrid does not set residual for out iterations
    if (updater.residual_ == nullptr)
    {
        dense_r_vec = parameters_.b->clone();

        auto one {gko::initialize<gko::matrix::Dense<scalar>>({1}, exec)};
        auto neg_one {gko::initialize<gko::matrix::Dense<scalar>>({-1}, exec)};
        parameters_.gkomatrix->apply(neg_one, updater.solution_, one, dense_r_vec);
    }

    const dist_vec* dense_r =
        (updater.residual_ == nullptr) ? dense_r_vec.get() : gko::as<dist_vec>(updater.residual_);

    auto norm1 = vec::create(exec, gko::dim<2> {1});
    dense_r->compute_norm1(norm1.get());
    auto norm1_host = vec::create(exec->get_master(), gko::dim<2> {1});
    norm1_host->copy_from(norm1.get());
    scalar residual_norm = norm1_host->at(0);
    // if (residual_norm != residual_norm) {
    //     NF_
    //     FatalErrorInFunction
    //         << " Problem with residual norm detected: " << residual_norm
    //         << exit(FatalError);
    // }

    bool result = false;

    // Store initial residual
    if (*(parameters_.iter) == 0)
    {
        //
        if (eval_norm_factor_)
        {
            norm_factor_ = compute_normfactor_dist(
                exec, dense_r, parameters_.gkomatrix, parameters_.x, parameters_.b
            );
        }

        *(parameters_.init_residual_norm) = residual_norm / norm_factor_;
    }

    residual_norm /= norm_factor_;


    if (parameters_.export_res)
    {
        parameters_.residual_norms->at(*(parameters_.iter)) = residual_norm;
    }

    *(parameters_.residual_norm) = residual_norm;

    scalar init_residual = *(parameters_.init_residual_norm);

    // stop if maximum number of iterations was reached
    if (*(parameters_.iter) >= parameters_.maxIter)
    {
        result = true;
    }
    // check if absolute tolerance is hit
    if (residual_norm < parameters_.absolute_tolerance)
    {
        result = true;
    }
    // check if relative tolerance is hit
    if (parameters_.relative_tolerance > 0
        && residual_norm < parameters_.relative_tolerance * init_residual)
    {
        result = true;
    }

    if (result)
    {
        this->set_all_statuses(stoppingId, setFinalized, stop_status);
        *one_changed = true;
    }

    *(parameters_.iter) += 1;

    auto end_eval = std::chrono::steady_clock::now();
    *(parameters_.time) =
        std::chrono::duration_cast<std::chrono::microseconds>(end_eval - start_eval).count() / 1.0;
    // std::cout << __FILE__ << "time " << *(parameters_.time) << " [mu s]\n";
    return result;
}

#endif

} // namespace Foam
