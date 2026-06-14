// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include <sstream>

#include "NeoN/linearAlgebra/ginkgo.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"

gko::config::pnode NeoN::la::ginkgo::parse(const Dictionary& dictIn)
{
    Dictionary dict = dictIn;
    // Remove the 'solver Ginkgo;' marker. Guard the string cast with isType:
    // 'solver' is not always a string — Ginkgo's solver::Ir keys its inner
    // operator as a sub-dictionary 'solver' (see makeJacobiSmoother in the
    // multigrid mapping), and parse() recurses into it. Casting that Dictionary
    // to std::string would throw bad_any_cast. A non-string 'solver' is a nested
    // factory and is parsed by the key loop below.
    if (dict.contains("solver") && dict.isType<std::string>("solver")
        && std::any_cast<std::string>(dict["solver"]) == "Ginkgo")
    {
        dict.remove("solver");
    }

    if (dict.contains("coupled"))
    {
        dict.remove("coupled");
    }

    // 'reportName' is a human-readable solver label (e.g. DICPCG) carried for
    // residual reporting; it is not a Ginkgo config key.
    if (dict.contains("reportName"))
    {
        dict.remove("reportName");
    }

    // 'l1ScaledResidual' opts into the L1-scaled residual stopping criterion
    // (handled separately via readL1ResidualControl); it is not a Ginkgo config key.
    if (dict.contains("l1ScaledResidual"))
    {
        dict.remove("l1ScaledResidual");
    }

    // 'checkFrequency' controls how often the L1 criterion evaluates the true residual
    // (handled via readL1ResidualControl); it is not a Ginkgo config key.
    if (dict.contains("checkFrequency"))
    {
        dict.remove("checkFrequency");
    }

    // 'minIter' / 'minIterFactor' steer the L1 criterion's minimum iteration count
    // (handled via readL1ResidualControl and the NeoFOAM PDESolver minIter steering);
    // they are not Ginkgo config keys.
    for (const auto& key : {std::string("minIter"), std::string("minIterFactor")})
    {
        if (dict.contains(key))
        {
            dict.remove(key);
        }
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
                if constexpr (std::is_same_v<value_type, float>)
                    return gko::config::pnode(static_cast<double>(dict.get<value_type>(key)));
                else
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
        // bool must be probed before int: Ginkgo config keys such as Pgm's
        // 'deterministic' or Multigrid's 'post_uses_pre' need a boolean pnode
        // (get_value<bool> calls pnode::get_boolean(), which throws on an integer
        // node). pnode's non-template bool constructor wins overload resolution
        // over the integral template, so this yields tag_t::boolean as required.
        if (auto node = parseAny(bool {}))
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
#elif defined(KOKKOS_ENABLE_SYCL)
                return gko::DpcppExecutor::create(
                    Kokkos::device_id(), gko::ReferenceExecutor::create()
                );
#endif
                throw std::runtime_error {"No valid GPU executor mapping available"};
            }
            else
            {
                throw std::runtime_error {"Unsupported executor type"};
            }
            return gko::ReferenceExecutor::create();
        },
        exec
    );
}


namespace NeoN::la::ginkgo
{

label computeNRows(const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys)
{
    return 3 * sys.rhs().size();
}

label computeNRows(const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& sys)
{
    return sys.rhs().size();
}

/*@brief create a new array by copying from view into ptr*/
template<typename T>
auto gkoCopyArray(std::shared_ptr<const gko::Executor> exec, std::span<T> values)
{
    return gko::make_const_array_view(exec, values.size(), values.data()).copy_to_array();
}

/* @brief create a ginkgo csr matrix by creating views into Csr<scalar> avoiding copies */
template<typename IndexType>
std::shared_ptr<const gko::LinOp>
createGkoMtxImpl(std::shared_ptr<const gko::Executor> exec, const CSRMatrix<scalar, IndexType>& mtx)
{
    const auto [coeffsV, sparsityV] = mtx.view();

    // NOTE we get a const view of the system but need a non const view to vals and indices
    auto vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(coeffsV.size()), coeffsV.data()
    );
    auto col = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(sparsityV.colIdxs.size()), sparsityV.colIdxs.data()
    );
    auto row = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(sparsityV.rowOffs.size()), sparsityV.rowOffs.data()
    );

    auto nrows = static_cast<gko::size_type>(mtx.nRows());
    return gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
        exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
    ));
}

template<typename IndexType>
std::shared_ptr<const gko::LinOp>
createGkoMtxImpl(std::shared_ptr<const gko::Executor> exec, const COOMatrix<scalar, IndexType>& mtx)
{
    const auto [coeffsV, sparsityV] = mtx.view();

    // NOTE we get a const view of the system but need a non const view to vals and indices
    auto vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(coeffsV.size()), coeffsV.data()
    );
    auto col = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(sparsityV.colIdxs.size()), sparsityV.colIdxs.data()
    );
    // sparsityV.rowOffs holds COO per-entry row indices; Ginkgo Csr::create_const needs
    // CSR row offsets (size nRows+1), which live in CooSparsityPattern::rowOffs_.
    const auto& csrRowOffs = mtx.rowOffs();
    auto row = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(csrRowOffs.size()), csrRowOffs.view().data()
    );

    auto nrows = static_cast<gko::size_type>(csrRowOffs.size() - 1);
    return gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
        exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
    ));
}

template<typename IndexType>
std::shared_ptr<const gko::LinOp>
createGkoMtxImpl(std::shared_ptr<const gko::Executor> exec, const CSRMatrix<Vec3, IndexType>& mtx)
{
    const auto rowsCopy = unpackRowOffs(mtx.rowOffs());
    const auto colsCopy = unpackColIdx(mtx.colIdxs(), rowsCopy, mtx.rowOffs());
    const auto valuesCopy = unpackMtxValues(mtx.values(), mtx.rowOffs(), rowsCopy);

    auto nrows = static_cast<gko::size_type>(3 * mtx.nRows());
    return gko::share(gko::matrix::Csr<scalar, IndexType>::create(
        exec,
        gko::dim<2> {nrows, nrows},
        gkoCopyArray(exec, valuesCopy.view()),
        gkoCopyArray(exec, colsCopy.view()),
        gkoCopyArray(exec, rowsCopy.view())
    ));
}

template<typename IndexType>
std::shared_ptr<const gko::LinOp>
createGkoMtxImpl(std::shared_ptr<const gko::Executor>, const COOMatrix<Vec3, IndexType>&)
{
    NF_THROW("createGkoMtxImpl: COOMatrix<Vec3> is not supported");
}

template<typename NeoNMatrixType>
std::shared_ptr<const gko::LinOp> createGkoMtx(const NeoNMatrixType& mtx)
{
    auto exec = getGkoExecutor(mtx.exec());
    return createGkoMtxImpl(exec, mtx);
}


template<typename VectorType>
SolverStatsEntry solve_impl(
    std::shared_ptr<const gko::Executor> exec,
    const VectorType& rhs,
    VectorType& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    std::unique_ptr<gko::LinOp> solver,
    const L1ResidualControl* l1Control = nullptr
)
{
    exec->synchronize();
    auto startEval = std::chrono::steady_clock::now();

    using vec = gko::matrix::Dense<scalar>;
    label nrows = rhs.size();
    const auto b = gkoVecView(exec, rhs.data(), nrows);
    auto x = gkoVecView(exec, xIn.data(), nrows);

    // L1-scaled residual path: stop and report on the scaled residual.
    if (l1Control != nullptr)
    {
        auto l1Res = solveWithL1Stop(exec, mtx, b, x, solver.get(), *l1Control);
        exec->synchronize();
        auto endEval = std::chrono::steady_clock::now();
        auto duration =
            static_cast<scalar>(
                std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
            )
            / 1000.0;
        return {static_cast<label>(l1Res.numIter), l1Res.initResNorm, l1Res.finalResNorm, duration};
    }

    // copy of rhs to compute the initial residual inline
    auto rhsCopy = VectorType(rhs);
    auto res = gkoVecView(exec, rhsCopy.data(), nrows);

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    mtx->apply(one, x, neg_one, res);

    auto init = gko::initialize<vec>({0.0}, exec);
    res->compute_norm2(init);
    scalar initResNorm = retrieve(init);

    std::shared_ptr<const gko::log::Convergence<scalar>> logger =
        gko::log::Convergence<scalar>::create();
    solver->add_logger(logger);
    solver->apply(b, x);

    scalar finalResNorm = retrieve(gko::as<vec>(logger->get_residual_norm()));
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

SolverStats solve_impl(
    std::shared_ptr<const gko::Executor> exec,
    const Vector<Vec3>& rhs,
    Vector<Vec3>& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    std::unique_ptr<gko::LinOp> solver
)
{
    exec->synchronize();
    auto startEval = std::chrono::steady_clock::now();

    using vec = gko::matrix::Dense<scalar>;
    label nrows = rhs.size();
    const auto b = gkoVecView(exec, rhs.data(), nrows); // [nrows x 3]
    auto x = gkoVecView(exec, xIn.data(), nrows);       // [nrows x 3]

    auto rhsCopy = Vector<Vec3>(rhs);
    auto res = gkoVecView(exec, rhsCopy.data(), nrows);

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    mtx->apply(one, x, neg_one, res);

    // compute_norm2 on [nrows x 3] writes a [1 x 3] result — one L2 norm per column.
    auto colNorms = [&](std::shared_ptr<gko::matrix::Dense<scalar>> v) -> std::array<scalar, 3>
    {
        auto nv = vec::create(exec, gko::dim<2> {1, 3});
        v->compute_norm2(nv);
        auto nh = vec::create(exec->get_master(), gko::dim<2> {1, 3});
        nh->copy_from(nv);
        return {nh->at(0, 0), nh->at(0, 1), nh->at(0, 2)};
    };
    auto initNorms = colNorms(res);

    std::shared_ptr<const gko::log::Convergence<scalar>> logger =
        gko::log::Convergence<scalar>::create();
    solver->add_logger(logger);
    solver->apply(b, x);

    auto rhsCopyFinal = Vector<Vec3>(rhs);
    auto resFinal = gkoVecView(exec, rhsCopyFinal.data(), nrows);
    mtx->apply(one, x, neg_one, resFinal);
    auto finalNorms = colNorms(resFinal);

    auto numIter = label(logger->get_num_iterations());
    exec->synchronize();
    auto endEval = std::chrono::steady_clock::now();
    auto duration =
        static_cast<scalar>(
            std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
        )
        / 1000.0;

    SolverStats stats;
    for (int i = 0; i < 3; ++i)
        stats.entries.push_back({numIter, initNorms[i], finalNorms[i], duration});
    return stats;
}


SolverStats GinkgoSolver::solve(
    const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
) const
{
    auto gkoMtx = createGkoMtx(sys.matrix());
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    return {solve_impl(gkoExec_, sys.rhs(), x, gkoMtx, factory_->generate(gkoMtx), l1Control)};
}

/* @brief create a ginkgo csr matrix by unpacking and copying the Csr<Vec3> input */
template<typename IndexType>
std::shared_ptr<const gko::matrix::Csr<scalar, IndexType>> createGkoMtxImpl(
    std::shared_ptr<const gko::Executor> exec,
    const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, IndexType>>& sys
)
{
    // NOTE we get a const view of the system but need a non const view to vals and indices
    const auto mtx = sys.matrix();
    const auto rowsCopy = unpackRowOffs(mtx.rowOffs());
    const auto colsCopy = unpackColIdx(mtx.colIdxs(), rowsCopy, mtx.rowOffs());
    const auto valuesCopy = unpackMtxValues(mtx.values(), mtx.rowOffs(), rowsCopy);

    auto nrows = static_cast<gko::size_type>(computeNRows(sys));
    return gko::share(gko::matrix::Csr<scalar, IndexType>::create(
        exec,
        gko::dim<2> {nrows, nrows},
        gkoCopyArray(exec, valuesCopy.view()),
        gkoCopyArray(exec, colsCopy.view()),
        gkoCopyArray(exec, rowsCopy.view())
    ));
}

// wrapper to solve a single component of a <vec3> equation
template<unsigned int I>
void solveComponent(
    auto& sys, auto& x, auto& exec, auto& factory, auto& stats, const L1ResidualControl* l1Control
)
{
    auto rhs = getComponent<I>(sys.rhs());
    auto xcopy = getComponent<I>(x);
    auto values = getComponent<I>(sys.matrix().values());
    auto sparsity = sys.matrix().sparsity();
    auto mtx = CSRMatrix<scalar, localIdx> {values, sparsity};
    auto gkoMtx = createGkoMtx(mtx);
    stats.entries.push_back(
        solve_impl(exec, rhs, xcopy, gkoMtx, factory->generate(gkoMtx), l1Control)
    );
    setComponent<I>(xcopy, x);
}

SolverStats GinkgoSolver::solve(
    const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
) const
{
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    if (coupled_)
    {
        const auto gkoMtx = createGkoMtx(sys.matrix());
        auto rhsCopy = unpackVecValues(sys.rhs());
        auto xCopy = unpackVecValues(x);

        auto stats =
            solve_impl(gkoExec_, rhsCopy, xCopy, gkoMtx, factory_->generate(gkoMtx), l1Control);

        packVecValues(xCopy, x);
        return {stats};
    }
    else
    {
        auto stats = SolverStats {};
        solveComponent<0>(sys, x, gkoExec_, factory_, stats, l1Control);
        solveComponent<1>(sys, x, gkoExec_, factory_, stats, l1Control);
        solveComponent<2>(sys, x, gkoExec_, factory_, stats, l1Control);
        return stats;
    }
}


SolverStats GinkgoSolver::solve(
    const LinearSystem<scalar, Vec3, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>>& sys,
    Vector<Vec3>& x
) const
{
    auto gkoMtx = createGkoMtx(sys.matrix());
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    if (l1Control)
    {
        gkoExec_->synchronize();
        auto startEval = std::chrono::steady_clock::now();

        label nrows = sys.rhs().size();
        const auto b = gkoVecView(gkoExec_, sys.rhs().data(), nrows); // [nrows x 3]
        auto xView = gkoVecView(gkoExec_, x.data(), nrows);           // [nrows x 3]
        auto solver = factory_->generate(gkoMtx);
        auto l1Res = solveWithL1Stop(gkoExec_, gkoMtx, b, xView, solver.get(), *l1Control);

        gkoExec_->synchronize();
        auto endEval = std::chrono::steady_clock::now();
        auto duration =
            static_cast<scalar>(
                std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
            )
            / 1000.0;

        SolverStats stats;
        for (int i = 0; i < 3; ++i)
            stats.entries.push_back(
                {l1Res.numIter, l1Res.perColInitNorms[i], l1Res.perColFinalNorms[i], duration}
            );
        return stats;
    }
    using vec = gko::matrix::Dense<scalar>;
    label nrows = sys.rhs().size();
    // Mutable [nrows x 3] Dense view of x — write-through to x.data()
    auto xDense = gkoVecView(gkoExec_, x.data(), nrows);
    const scalar* rhsScalar = reinterpret_cast<const scalar*>(sys.rhs().data());

    gkoExec_->synchronize();
    SolverStats stats;
    for (gko::size_type col = 0; col < 3; ++col)
    {
        auto t0 = std::chrono::steady_clock::now();
        gko::span spanAll {gko::size_type {0}, static_cast<gko::size_type>(nrows)};
        gko::span spanC {col, col + 1};

        // [nrows x 1] const strided Dense view of rhs column col (no copy)
        // Data at rhsScalar+col, stride 3; array size = 3*nrows-col covers all nrows elements
        auto b_col = gko::share(vec::create_const(
            gkoExec_,
            gko::dim<2> {static_cast<gko::size_type>(nrows), 1},
            gko::array<scalar>::const_view(
                gkoExec_, static_cast<gko::size_type>(3 * nrows - col), rhsScalar + col
            ),
            3
        ));

        // [nrows x 1] mutable strided Dense view of x column col — write-through to x.data()
        auto x_col = xDense->create_submatrix(spanAll, spanC);

        auto initNorm_v = vec::create(gkoExec_, gko::dim<2> {1, 1});
        b_col->compute_norm2(initNorm_v); // initResNorm = ||b_col||₂  (x starts at 0)
        scalar initResNorm = retrieve(initNorm_v);

        auto solver = factory_->generate(gkoMtx);
        std::shared_ptr<const gko::log::Convergence<scalar>> logger =
            gko::log::Convergence<scalar>::create();
        solver->add_logger(logger);
        solver->apply(b_col, x_col);

        scalar finalResNorm = retrieve(gko::as<vec>(logger->get_residual_norm()));
        auto numIter = label(logger->get_num_iterations());
        gkoExec_->synchronize();
        auto duration = static_cast<scalar>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::steady_clock::now() - t0
                        )
                                                .count())
                      / 1000.0;
        stats.entries.push_back({numIter, initResNorm, finalResNorm, duration});
    }
    return stats;
}

template std::shared_ptr<const gko::LinOp>
createGkoMtx<CSRMatrix<scalar, localIdx>>(const CSRMatrix<scalar, localIdx>&);

template std::shared_ptr<const gko::LinOp>
createGkoMtx<COOMatrix<scalar, localIdx>>(const COOMatrix<scalar, localIdx>&);

}

#endif
