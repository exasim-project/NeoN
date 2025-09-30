// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include <sstream>

#include "NeoN/linearAlgebra/ginkgo.hpp"

gko::config::pnode NeoN::la::ginkgo::parse(const Dictionary& dictIn)
{
    Dictionary dict = dictIn;
    // remove 'solver Ginkgo;' entry
    if (dict.contains("solver") && std::any_cast<std::string>(dict["solver"]) == "Ginkgo")
    {
        dict.remove("solver");
    }

    // check if an external file name is given
    if (dict.contains("configFile"))
    {

        std::string fn_str {};
        auto fn = dict["configFile"];
        if (fn.type() == typeid(std::string))
        {
            fn_str = std::any_cast<std::string>(fn);
        }
        else
        {
            auto token = std::any_cast<TokenList>(fn);
            std::stringstream s;
            for (NeoN::size_t i = 0; i < token.size() - 1; i++)
            {
                s << token.next<std::string>() << "/";
            }
            s << token.next<std::string>();
            fn_str = s.str();
        }

        return gko::ext::config::parse_json_file(fn_str);
    }

    auto parseData = [&](auto key)
    {
        auto parseAny = [&](auto blueprint)
        {
            using value_type = decltype(blueprint);
            if (dict[key].type() == typeid(value_type))
            {
                return gko::config::pnode(dict.get<value_type>(key));
            }
            else
            {
                return gko::config::pnode();
            }
        };

        if (auto node = parseAny(std::string()))
        {
            return node;
        }
        if (auto node = parseAny(static_cast<const char*>(nullptr)))
        {
            return node;
        }
        if (auto node = parseAny(int {}))
        {
            return node;
        }
        if (auto node = parseAny(static_cast<unsigned int>(0)))
        {
            return node;
        }
        if (auto node = parseAny(double {}))
        {
            return node;
        }
        if (auto node = parseAny(float {}))
        {
            return node;
        }

        NF_THROW("Dictionary key " + key + " has unsupported type: " + dict[key].type().name());
    };
    gko::config::pnode::map_type result;
    for (const auto& key : dict.keys())
    {
        gko::config::pnode node;
        if (dict.isDict(key))
        {
            node = parse(dict.subDict(key));
        }
        else
        {
            node = parseData(key);
        }
        result.emplace(key, node);
    }
    return gko::config::pnode {result};
}


// TODO: check if this can be replaced by Ginkgos executor mapping
std::shared_ptr<gko::Executor> NeoN::la::ginkgo::getGkoExecutor(NeoN::Executor exec)
{
    return std::visit(
        [](auto concreteExec) -> std::shared_ptr<gko::Executor>
        {
            using ExecType = std::decay_t<decltype(concreteExec)>;
            if constexpr (std::is_same_v<ExecType, NeoN::SerialExecutor>)
            {
                return gko::ReferenceExecutor::create();
            }
            else if constexpr (std::is_same_v<ExecType, NeoN::CPUExecutor>)
            {
#if defined(KOKKOS_ENABLE_OMP)
                return gko::OmpExecutor::create();
#elif defined(KOKKOS_ENABLE_THREADS)
                return gko::ReferenceExecutor::create();
#endif
            }
            else if constexpr (std::is_same_v<ExecType, NeoN::GPUExecutor>)
            {
#if defined(KOKKOS_ENABLE_CUDA)
                return gko::CudaExecutor::create(
                    Kokkos::device_id(), gko::ReferenceExecutor::create()
                );
#elif defined(KOKKOS_ENABLE_HIP)
                return gko::HipExecutor::create(
                    Kokkos::device_id(), gko::ReferenceExecutor::create()
                );
#endif
                throw std::runtime_error("No valid GPU executor mapping available");
            }
            else
            {
                throw std::runtime_error("Unsupported executor type");
            }
            return gko::ReferenceExecutor::create();
        },
        exec
    );
}


namespace NeoN::la::ginkgo
{

template<typename IndexType>
std::shared_ptr<gko::matrix::Csr<scalar, IndexType>>
createGkoMtx(std::shared_ptr<const gko::Executor> exec, const LinearSystem<scalar, IndexType>& sys)
{
    auto nrows = static_cast<gko::dim<2>::dimension_type>(sys.rhs().size());
    auto mtx = sys.view().matrix;
    // NOTE we get a const view of the system but need a non const view to vals and indices
    // auto vals = createConstGkoArray(exec, mtx.values).copy_to_array();
    auto vals = gko::array<scalar>::view(
        exec, static_cast<gko::size_type>(mtx.values.size()), const_cast<scalar*>(mtx.values.data())
    );
    // auto col = createGkoArray(exec, mtx.colIdxs);
    auto col = gko::array<IndexType>::view(
        exec,
        static_cast<gko::size_type>(mtx.colIdxs.size()),
        const_cast<IndexType*>(mtx.colIdxs.data())
    );
    // auto row = createGkoArray(exec, mtx.rowOffs);
    auto row = gko::array<IndexType>::view(
        exec,
        static_cast<gko::size_type>(mtx.rowOffs.size()),
        const_cast<IndexType*>(mtx.rowOffs.data())
    );
    return gko::share(gko::matrix::Csr<scalar, IndexType>::create(
        exec, gko::dim<2> {nrows, nrows}, vals, col, row
    ));
}


SolverStats GinkgoSolver::solve(const LinearSystem<scalar, localIdx>& sys, Vector<scalar>& x) const
{
    auto startEval = std::chrono::steady_clock::now();
    using vec = gko::matrix::Dense<scalar>;

    auto retrieve = [](const auto& in)
    {
        auto host = vec::create(in->get_executor()->get_master(), gko::dim<2> {1});
        scalar res = host->copy_from(in)->at(0);
        return res;
    };

    auto nrows = sys.rhs().size();

    auto gkoMtx = createGkoMtx(gkoExec_, sys);
    auto solver = factory_->generate(gkoMtx);

    std::shared_ptr<const gko::log::Convergence<scalar>> logger =
        gko::log::Convergence<scalar>::create();
    solver->add_logger(logger);

    auto rhs = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
    auto rhs2 = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
    auto gkoX = detail::createGkoDense(gkoExec_, x.data(), nrows);

    auto one = gko::initialize<vec>({1.0}, gkoExec_);
    auto neg_one = gko::initialize<vec>({-1.0}, gkoExec_);
    auto init = gko::initialize<vec>({0.0}, gkoExec_);
    gkoMtx->apply(one, gkoX, neg_one, rhs2);
    rhs->compute_norm2(init);
    scalar initResNorm = retrieve(init);

    solver->apply(rhs, gkoX);

    scalar finalResNorm = retrieve(gko::as<vec>(logger->get_residual_norm()));

    auto numIter = label(logger->get_num_iterations());

    auto endEval = std::chrono::steady_clock::now();
    auto duration =
        static_cast<scalar>(
            std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
        )
        / 1000.0;
    return {numIter, initResNorm, finalResNorm, duration};
}

template<typename IndexType>
std::shared_ptr<gko::matrix::Csr<Vec3, IndexType>>
createGkoMtx(std::shared_ptr<const gko::Executor> exec, const LinearSystem<Vec3, IndexType>& sys)
{
    auto nrows = static_cast<gko::dim<2>::dimension_type>(sys.rhs().size());
    auto mtx = sys.view().matrix;
    // NOTE we get a const view of the system but need a non const view to vals and indices
    // auto vals = createConstGkoArray(exec, mtx.values).copy_to_array();
    // auto vals = gko::array<Vec3>::view(
    //     exec,
    //     static_cast<gko::size_type>(mtx.values.size()),
    //     const_cast<Vec3*>(mtx.values.data())
    // );
    // // auto col = createGkoArray(exec, mtx.colIdxs);
    // auto col = gko::array<IndexType>::view(
    //     exec,
    //     static_cast<gko::size_type>(mtx.colIdxs.size()),
    //     const_cast<IndexType*>(mtx.colIdxs.data())
    // );
    // // auto row = createGkoArray(exec, mtx.rowOffs);
    // auto row = gko::array<IndexType>::view(
    //     exec,
    //     static_cast<gko::size_type>(mtx.rowOffs.size()),
    //     const_cast<IndexType*>(mtx.rowOffs.data())
    // );

    // return gko::share(gko::matrix::Csr<Vec3, IndexType>::create(
    //     exec, gko::dim<2> {nrows, nrows}, vals, col, row
    // ));
}

SolverStats GinkgoSolver::solve(const LinearSystem<Vec3, localIdx>& sys, Vector<Vec3>& x) const
{
    auto startEval = std::chrono::steady_clock::now();
    using vec = gko::matrix::Dense<scalar>;

    auto retrieve = [](const auto& in)
    {
        auto host = vec::create(in->get_executor()->get_master(), gko::dim<2> {1});
        scalar res = host->copy_from(in)->at(0);
        return res;
    };

    auto nrows = 3 * sys.rhs().size();

    auto gkoMtx = createGkoMtx(gkoExec_, sys);
    // auto solver = factory_->generate(gkoMtx);

    // std::shared_ptr<const gko::log::Convergence<scalar>> logger =
    //     gko::log::Convergence<scalar>::create();
    // solver->add_logger(logger);

    // auto rhs = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
    // auto rhs2 = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
    // auto gkoX = detail::createGkoDense(gkoExec_, x.data(), nrows);

    // auto one = gko::initialize<vec>({1.0}, gkoExec_);
    // auto neg_one = gko::initialize<vec>({-1.0}, gkoExec_);
    // auto init = gko::initialize<vec>({0.0}, gkoExec_);
    // gkoMtx->apply(one, gkoX, neg_one, rhs2);
    // rhs->compute_norm2(init);
    // scalar initResNorm = retrieve(init);

    // solver->apply(rhs, gkoX);

    // scalar finalResNorm = retrieve(gko::as<vec>(logger->get_residual_norm()));

    // auto numIter = label(logger->get_num_iterations());

    // auto endEval = std::chrono::steady_clock::now();
    // auto duration =
    //     static_cast<scalar>(
    //         std::chrono::duration_cast<std::chrono::microseconds>(endEval -
    //         startEval).count()
    //     )
    //     / 1000.0;
    // return {numIter, initResNorm, finalResNorm, duration};
}


}

#endif
