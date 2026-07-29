// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>

namespace blockamr::la
{

// Vectors are Dense on one rank and distributed::Vector on several -- unrelated
// types, so `gko::as<Dense>` throws on the distributed one; these accessors are
// the seam. Operators stay rank-local (the halo comes from AMReX FillBoundary,
// and everything they do to a vector is elementwise); the distributed type
// exists only so the dots/norms inside Ginkgo's Krylov solvers become
// allreduces. GINKGO_BUILD_MPI tracks NeoN_WITH_MPI; without it the distributed
// types do not exist and only the Dense branch remains.
#if GINKGO_BUILD_MPI
template<class V>
using DistVec = gko::experimental::distributed::Vector<V>;
#endif

// Writable raw pointer to the process-local values of x.
template<class V>
V* localValues(gko::LinOp* x)
{
#if GINKGO_BUILD_MPI
    if (auto* d = dynamic_cast<DistVec<V>*>(x))
    {
        return d->get_local_values();
    }
#endif
    return gko::as<gko::matrix::Dense<V>>(x)->get_values();
}

template<class V>
const V* localValues(const gko::LinOp* x)
{
#if GINKGO_BUILD_MPI
    if (const auto* d = dynamic_cast<const DistVec<V>*>(x))
    {
        return d->get_const_local_values();
    }
#endif
    return gko::as<const gko::matrix::Dense<V>>(x)->get_const_values();
}

// Process-local row count. NOT get_size()[0], which on a distributed vector is
// the global count -- sizing flat vectors by it is a past multi-rank bug.
inline gko::size_type localRows(const gko::LinOp* x)
{
#if GINKGO_BUILD_MPI
    if (const auto* d = dynamic_cast<const DistVec<double>*>(x))
    {
        return d->get_local_vector()->get_size()[0];
    }
    if (const auto* d = dynamic_cast<const DistVec<float>*>(x))
    {
        return d->get_local_vector()->get_size()[0];
    }
#endif
    return x->get_size()[0];
}

// Non-owning Dense VIEW of the process-local part, for callers needing a Dense
// object rather than a pointer. Aliases x: writing through the view writes x.
template<class V>
std::unique_ptr<gko::matrix::Dense<V>> localView(gko::LinOp* x)
{
    const auto n = localRows(x);
    auto exec = x->get_executor();
    return gko::matrix::Dense<V>::create(
        exec, gko::dim<2> {n, 1}, gko::make_array_view(exec, n, localValues<V>(x)), 1
    );
}

// const overload, for the host paths that clone b to the master executor.
template<class V>
std::unique_ptr<const gko::matrix::Dense<V>> localView(const gko::LinOp* x)
{
    const auto n = localRows(x);
    auto exec = x->get_executor();
    return gko::matrix::Dense<V>::create_const(
        exec, gko::dim<2> {n, 1}, gko::make_const_array_view(exec, n, localValues<V>(x)), 1
    );
}

// ||v||_2 and v.w over ALL ranks: Ginkgo's own compute_norm2/compute_dot carry
// the allreduce. The 1x1 result is staged through the host master for the caller.
inline double reduceScalar(const gko::LinOp* v, const gko::matrix::Dense<double>* r)
{
    return gko::clone(v->get_executor()->get_master(), r)->at(0, 0);
}

inline double globalNorm2(const gko::LinOp* v)
{
    auto exec = v->get_executor();
    auto nrm = gko::matrix::Dense<double>::create(exec, gko::dim<2> {1, 1});
#if GINKGO_BUILD_MPI
    if (const auto* d = dynamic_cast<const DistVec<double>*>(v))
    {
        d->compute_norm2(nrm);
        return reduceScalar(v, nrm.get());
    }
#endif
    gko::as<const gko::matrix::Dense<double>>(v)->compute_norm2(nrm);
    return reduceScalar(v, nrm.get());
}

inline double globalDot(const gko::LinOp* v, const gko::LinOp* w)
{
    auto exec = v->get_executor();
    auto res = gko::matrix::Dense<double>::create(exec, gko::dim<2> {1, 1});
#if GINKGO_BUILD_MPI
    if (const auto* d = dynamic_cast<const DistVec<double>*>(v))
    {
        d->compute_dot(w, res);
        return reduceScalar(v, res.get());
    }
#endif
    gko::as<const gko::matrix::Dense<double>>(v)->compute_dot(w, res);
    return reduceScalar(v, res.get());
}

// Ginkgo-side view of a rank-local Dense buffer: on >1 rank a distributed::Vector
// aliasing it (data not owned, so writes through the buffer are seen), which is
// what makes a solver's internal dots/norms allreduces; on one rank (or without
// MPI) a plain Dense view.
//
// Must use AMReX's communicator, not MPI_COMM_WORLD: the operators' halo exchange
// runs on ParallelContext::CommunicatorSub, and a reduction on another
// communicator would reduce over a different set of ranks.
//
// BUILD TRAP: two overloads rather than a template, defined in persistent.cpp --
// nvcc rejects the shared_ptr<Dense<V>> -> shared_ptr<LinOp> conversion in a
// TEMPLATE signature ("use of built-in trait __remove_cv in function
// signature"); the same conversion in a plain function compiles. double is the
// fp64 Krylov paths, float the mixed-precision GMG bottom solver.
std::shared_ptr<gko::LinOp> makeGlobalVec(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type nGlobal,
    gko::matrix::Dense<double>* local
);
std::shared_ptr<gko::LinOp> makeGlobalVec(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type nGlobal,
    gko::matrix::Dense<float>* local
);

} // namespace blockamr::la
