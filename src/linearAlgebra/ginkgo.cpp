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
#elif defined(KOKKOS_ENABLE_SYCL)
                return gko::DpcppExecutor::create(
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

label computeNRows(const LinearSystem<Vec3, localIdx>& sys) { return 3 * sys.rhs().size(); }

label computeNRows(const LinearSystem<scalar, localIdx>& sys) { return sys.rhs().size(); }

/*@brief create a array non const view into data given by ptr*/
template<typename T>
gko::array<T> gkoArrayView(std::shared_ptr<const gko::Executor> exec, std::span<T> values)
{
    return gko::make_array_view(exec, values.size(), values.data());
}

/*@brief create a new array by copying from view into ptr*/
template<typename T>
auto gkoCopyArray(std::shared_ptr<const gko::Executor> exec, std::span<T> values)
{
    return gko::make_const_array_view(exec, values.size(), values.data()).copy_to_array();
}


/*@brief create a dense non const view into data given by ptr*/
// std::shared_ptr<const gko::matrix::Dense<scalar>>
std::shared_ptr<gko::LinOp> gkoVecView(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    scalar* ptr,
    localIdx s
)
{
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    auto size = static_cast<std::size_t>(s);

    auto ret = gko::share(dist_vec::create(
        exec,
        comm,
        vec::create(exec, gko::dim<2> {size, 1}, gkoArrayView(exec, std::span {ptr, size}), 1)
    ));

    return ret;

    // return gko::share(gko::matrix::Dense<scalar>::create(
    //     exec, gko::dim<2> {size, 1}, gkoArrayView(exec, std::span {ptr, size}), 1
    // ));
}

void writeToDisk(std::string fn, std::shared_ptr<gko::LinOp> A)
{
    std::ofstream stream {fn};
    stream << std::setprecision(15);
    gko::write(stream, A.get());
}

/*@brief create a dense const view into data given by ptr*/
// std::shared_ptr<const gko::matrix::Dense<scalar>>
std::shared_ptr<gko::LinOp> gkoVecView(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const scalar* ptr,
    localIdx s
)
{
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using vec = gko::matrix::Dense<scalar>;

    auto size = static_cast<std::size_t>(s);

    auto ret = gko::share(dist_vec::create(
        exec,
        comm,
        vec::create(
            exec,
            gko::dim<2> {size, 1},
            gko::make_const_array_view(exec, size, ptr).copy_to_array(),
            1
        )
    ));
    // gkoArrayView(exec, std::span {ptr, size}), 1)));
    // gko::array<scalar>::const_view(exec, size, ptr), 1)));

    return ret;
    // return gko::share(gko::matrix::Dense<scalar>::create_const(
    //     exec, gko::dim<2> {size, 1}, gko::array<scalar>::const_view(exec, size, ptr), 1
    // ));
}


/* @brief create a ginkgo csr matrix by creating views into Csr<scalar> avoiding copies */
template<typename IndexType>
std::shared_ptr<const gko::LinOp> createGkoMtx(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const LinearSystem<scalar, IndexType>& sys
)
{
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, label>;

    const auto mtx = sys.view().matrix;
    // NOTE we get a const view of the system but need a non const view to vals and indices
    auto vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(mtx.values.size()), mtx.values.data()
    );
    auto col = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(mtx.colIdxs.size()), mtx.colIdxs.data()
    );
    auto row = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(mtx.rowOffs.size()), mtx.rowOffs.data()
    );

    auto nrows = static_cast<gko::size_type>(computeNRows(sys));

    // FIXME currently no communication with other rank
    auto partition =
        gko::share(gko::experimental::distributed::build_partition_from_local_size<label, label>(
            exec, comm, nrows
        ));

    // FIXME currently no communication with other rank
    // recv_connections, ie the send_idxs of the neighbouring ranks in global indexing
    auto recv_connections = gko::array<label>(exec, 0);

    auto imap = gko::experimental::distributed::index_map<label, label>(
        exec, partition, comm.rank(), recv_connections
    );

    std::shared_ptr<gko::LinOp> localMtx =
        gko::share(
            gko::matrix::Csr<scalar, IndexType>::create_const(
                exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
            )
        )
            ->clone();

    // writeToDisk("localA" + std::to_string(comm.rank()) + ".mtx", localMtx);

    std::shared_ptr<gko::LinOp> nonLocalMtx =
        gko::share(gko::matrix::Csr<scalar, IndexType>::create(exec, gko::dim<2> {nrows, 0}));
    // writeToDisk("nonLocalA" + std::to_string(comm.rank()) + ".mtx", nonLocalMtx);

    return gko::share(dist_mtx::create(exec, comm, imap, localMtx, nonLocalMtx));
}


/*@brief helper function to get a scalar dense value from a device back to the host*/
template<typename InType>
scalar retrieve(const InType& in)
{
    using vec = gko::matrix::Dense<scalar>;
    auto host = vec::create(in->get_executor()->get_master(), gko::dim<2> {1});
    return host->copy_from(in)->at(0);
};

SolverStats solve_impl(
    std::shared_ptr<const gko::Executor> exec,
    const Vector<scalar>& rhs,
    Vector<scalar>& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    std::unique_ptr<gko::LinOp> solver
)
{
    // FIXME dont re-init
    bool forceHostBuffer = false;
    mpi::Environment env;
    auto comm = gko::experimental::mpi::communicator(env.comm(), forceHostBuffer);

    auto startEval = std::chrono::steady_clock::now();

    using vec = gko::matrix::Dense<scalar>;
    label nrows = rhs.size();
    const auto b = gkoVecView(exec, comm, rhs.data(), nrows);
    auto x = gkoVecView(exec, comm, xIn.data(), nrows);

    // create a copy of rhs so that we can inline compute
    // the residual
    auto rhsCopy = Vector<scalar>(rhs);
    auto res = gkoVecView(exec, comm, rhsCopy.data(), nrows);

    // compute Ax-b -> res
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    mtx->apply(one, x, neg_one, res);

    // FIXME dont re-init
    auto init = gko::initialize<vec>({0.0}, exec);
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    gko::as<dist_vec>(res)->compute_norm2(init);
    scalar initResNorm = retrieve(init);

    std::shared_ptr<const gko::log::Convergence<scalar>> logger =
        gko::log::Convergence<scalar>::create();
    solver->add_logger(logger);
    solver->apply(b, x);

    // since we work on a copy we need to copy back
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


SolverStats GinkgoSolver::solve(const LinearSystem<scalar, localIdx>& sys, Vector<scalar>& x) const
{
    bool forceHostBuffer = false;
    mpi::Environment env;
    auto comm = gko::experimental::mpi::communicator(env.comm(), forceHostBuffer);

    auto gkoMtx = createGkoMtx(gkoExec_, comm, sys);
    auto solver = factory_->generate(gkoMtx);
    return solve_impl(gkoExec_, sys.rhs(), x, gkoMtx, std::move(solver));
}

/* @brief create a ginkgo csr matrix by unpacking and copying the Csr<Vec3> input */
template<typename IndexType>
std::shared_ptr<const gko::LinOp> createGkoMtx(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const LinearSystem<Vec3, IndexType>& sys
)
{
    auto nrows = static_cast<gko::size_type>(computeNRows(sys));
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, label>;

    // FIXME currently no communication with other rank
    auto partition =
        gko::share(gko::experimental::distributed::build_partition_from_local_size<label, label>(
            exec, comm, nrows
        ));

    // FIXME currently no communication with other rank
    // recv_connections, ie the send_idxs of the neighbouring ranks in global indexing
    auto recv_connections = gko::array<label>(exec, 0);

    auto imap = gko::experimental::distributed::index_map<label, label>(
        exec, partition, comm.rank(), recv_connections
    );

    // NOTE we get a const view of the system but need a non const view to vals and indices
    const auto mtx = sys.matrix();
    const auto rowsCopy = unpackRowOffs(mtx.rowOffs());
    const auto colsCopy = unpackColIdx(mtx.colIdxs(), rowsCopy, mtx.rowOffs());
    const auto valuesCopy = unpackMtxValues(mtx.values(), mtx.rowOffs(), rowsCopy);
    auto localMtx = gko::share(gko::matrix::Csr<scalar, IndexType>::create(
        exec,
        gko::dim<2> {nrows, nrows},
        gkoCopyArray(exec, valuesCopy.view()),
        gkoCopyArray(exec, colsCopy.view()),
        gkoCopyArray(exec, rowsCopy.view())
    ));
    // FIXME currently only empty nonLocalMtx
    auto nonLocalMtx =
        gko::share(gko::matrix::Csr<scalar, IndexType>::create(exec, gko::dim<2> {nrows, 0}));

    return gko::share(dist_mtx::create(exec, comm, imap, localMtx, nonLocalMtx));
}

SolverStats GinkgoSolver::solve(const LinearSystem<Vec3, localIdx>& sys, Vector<Vec3>& x) const
{
    // auto environment = sys.environment();
    bool forceHostBuffer = false;

    // FIXME dont re-init
    mpi::Environment env;
    auto comm = gko::experimental::mpi::communicator(env.comm(), forceHostBuffer);

    const auto gkoMtx = createGkoMtx(gkoExec_, comm, sys);
    auto solver = factory_->generate(gkoMtx);

    auto rhsCopy = unpackVecValues(sys.rhs());
    auto xCopy = unpackVecValues(x);

    auto stats = solve_impl(gkoExec_, rhsCopy, xCopy, gkoMtx, std::move(solver));

    packVecValues(xCopy, x);
    return stats;
}


}

#endif
