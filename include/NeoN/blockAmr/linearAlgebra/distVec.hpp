// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>

namespace blockamr::la
{

// The vectors these operators are applied to are Dense on one rank and
// gko::experimental::distributed::Vector on several. The two are unrelated
// types, so `gko::as<Dense>` -- which every operator here used to do -- throws
// on the distributed one; these three accessors are the seam.
//
// Only the vectors Ginkgo OWNS differ. The operators themselves stay entirely
// rank-local: the halo comes from AMReX's FillBoundary, not from Ginkgo, and
// everything an operator does to a vector (gather, scatter, scale, add_scaled)
// is elementwise. What actually needs the distributed type is the dot products
// and norms INSIDE Ginkgo's Krylov solvers, which have to reduce across ranks;
// making the solver's b/x distributed makes its cloned work vectors distributed
// too, and those reductions become allreduces with no further help from us.
//
// So: an operator asks for the process-local part and is none the wiser. The
// local part of a Dense is the whole of it.
//
// GINKGO_BUILD_MPI tracks NeoN_WITH_MPI; without it the distributed types do
// not exist and the Dense branch is the only one.
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

// Process-local row count. NOT get_size()[0] on a distributed vector -- that is
// the global count, and the whole 5b bug was flat vectors sized by it.
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

// Non-owning Dense VIEW of the process-local part, for the few places that need
// a Dense object rather than a pointer (scale, add_scaled, convert_to). It
// aliases x's memory -- writing through the view writes x.
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

// ||v||_2 and v.w over ALL ranks. On a distributed vector Ginkgo's own
// compute_norm2/compute_dot already carry the allreduce; on a Dense there is
// only one rank's worth to reduce. The 1x1 result is staged through the host
// master because the caller wants a plain double.
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

// Ginkgo-side view of a rank-local Dense buffer.
//
// On >1 rank: a distributed::Vector aliasing the buffer, whose global size is
// the total row count. It owns the Dense object but not the data, so writing
// through the original buffer writes the vector. Handing THIS to a Ginkgo solver
// is what makes the solver's internal dots and norms allreduces, because its
// work vectors are clones of it.
//
// On one rank (or a build without MPI, where the distributed types do not exist)
// there is nothing to reduce and a plain Dense view comes back, keeping the
// whole distributed machinery out of the single-rank path.
//
// AMReX's communicator, not MPI_COMM_WORLD: the halo exchange inside the
// operators runs on ParallelContext::CommunicatorSub, and a reduction taken on a
// different communicator would be reducing over a different set of ranks.
//
// Two overloads rather than a template, and defined in persistent.cpp rather
// than here: nvcc rejects the shared_ptr<Dense<V>> -> shared_ptr<LinOp>
// conversion when it appears in a TEMPLATE signature ("use of built-in trait
// __remove_cv in function signature"). The same conversion in a plain function
// compiles. double is the fp64 Krylov paths; float is the mixed-precision GMG
// hierarchy's bottom solver.
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
