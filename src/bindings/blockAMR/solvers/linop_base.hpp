// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_ParallelContext.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>

#include "profiling.hpp"
#include "types.hpp"

namespace blockamr::solvers
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
// On >1 rank: a distributed::Vector aliasing `local`'s memory, whose global size
// is the total row count. It owns no data -- writing through `local` writes the
// vector. Handing THIS to a Ginkgo solver is what makes the solver's internal
// dots and norms allreduces, because its work vectors are clones of it.
//
// On one rank (or a build without MPI, where the distributed types do not exist)
// there is nothing to reduce, so the Dense is returned unchanged and the whole
// distributed machinery stays out of the single-rank path.
//
// AMReX's communicator, not MPI_COMM_WORLD: the halo exchange inside the
// operators runs on ParallelContext::CommunicatorSub, and a reduction taken on a
// different communicator would be reducing over a different set of ranks.
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

// CRTP base for the matrix-free Ginkgo operators in this directory. It bundles
// the two mixins every one of them needs (gko::EnableLinOp for the LinOp
// plumbing, gko::EnableCreateMethod for the static create()) and supplies the
// ONE implementation of the "advanced" apply_impl:
//   x = alpha * op(b) + beta * x,
// expressed through the derived class' simple apply_impl(b, x) on a temporary.
// Every derived operator previously carried a byte-identical copy of it.
//
// A derived class D:
//   - derives as `public AmrexLinOpBase<D>` (in place of the former
//     `public gko::EnableLinOp<D>, public gko::EnableCreateMethod<D>`),
//   - forwards to `AmrexLinOpBase<D>(exec)` / `AmrexLinOpBase<D>(exec, size)`
//     in its constructors (in place of `gko::EnableLinOp<D>(...)`),
//   - implements only `apply_impl(const gko::LinOp* b, gko::LinOp* x) const`,
//     preceded by `using AmrexLinOpBase<D>::apply_impl;` so that declaration
//     does not hide the advanced overload (nvcc warning 611 / -Woverloaded-
//     virtual; the code is correct either way, the using-declaration only
//     keeps the build log clean).
// The exec-only constructor stays required by the polymorphic-object machinery
// (create_default / clear), which does `new D(exec)`.
// V is the value type of the Dense vectors this operator is applied to -- double
// for the fp64 Krylov path, float for the mixed-precision one. It appears only in
// the Dense casts below; gko::EnableLinOp carries no value type, so a derived
// operator is a plain gko::LinOp either way and Cg<float> accepts it directly.
template<class D, class V = double>
class AmrexLinOpBase : public gko::EnableLinOp<D>, public gko::EnableCreateMethod<D>
{
protected:

    using DenseV = gko::matrix::Dense<V>;


    explicit AmrexLinOpBase(std::shared_ptr<const gko::Executor> exec) : gko::EnableLinOp<D>(exec)
    {}

    AmrexLinOpBase(std::shared_ptr<const gko::Executor> exec, const gko::dim<2>& size)
        : gko::EnableLinOp<D>(exec, size)
    {}

    // Supplied by the derived class; re-declared here so it stays visible in
    // this scope for the advanced overload below to call.
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override = 0;

    // x = alpha * op(b) + beta * x.
    //
    // Every derived apply_impl(b, x) OVERWRITES the whole of x — each one ends
    // in a gather across the entire flat vector — so the intermediate needs no
    // initial contents. It is therefore a reused scratch buffer rather than a
    // clone of x: cloning cost an allocation plus a full device copy of x per
    // call, and both were discarded by the very next line.
    //
    // The beta branches skip whole vector passes. beta == 1 is the case Ginkgo's
    // Ir and the Krylov initial residual take (r = b - A x), where scale(beta)
    // is a read-modify-write pass that multiplies by one. beta == 0 discards x
    // entirely, so op(b) can be written straight into it with no scratch at all.
    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        prof::Timer tAll("adv.apply");
        // A view, so scale/add_scaled below hit x's own memory. Both are
        // elementwise, so doing them on the local part is the whole operation.
        auto denseX = localView<V>(x);
        const double alphaVal = hostScalar(alpha);
        const double betaVal = hostScalar(beta);

        if (betaVal == 0.0)
        {
            this->apply_impl(b, x);
            if (alphaVal != 1.0)
            {
                prof::Timer t("adv.scale");
                denseX->scale(alpha);
            }
            return;
        }

        const gko::dim<2> size = denseX->get_size();
        if (!scratch_ || scratch_->get_size() != size)
        {
            prof::Timer t("adv.alloc");
            scratch_ = DenseV::create(this->get_executor(), size);
        }
        this->apply_impl(b, scratch_.get());
        if (betaVal != 1.0)
        {
            prof::Timer t("adv.scale");
            denseX->scale(beta);
        }
        {
            prof::Timer t("adv.addscaled");
            denseX->add_scaled(alpha, scratch_);
        }
    }

private:

    // alpha/beta are 1x1 Dense on the solve executor; a device value is staged
    // through the host master to read it (cf. ResidualHistoryLogger::readScalar).
    static double hostScalar(const gko::LinOp* s)
    {
        auto d = gko::as<DenseV>(s);
        auto exec = d->get_executor();
        if (exec->get_master().get() != exec.get())
        {
            return gko::clone(exec->get_master(), d)->at(0, 0);
        }
        return d->at(0, 0);
    }

    // shared_ptr, not unique_ptr: Ginkgo's EnablePolymorphicAssignment gives
    // these operators a copy-assignment, which a move-only member would delete.
    // Sharing a scratch buffer between copies is harmless — it holds no state
    // across calls.
    mutable std::shared_ptr<DenseV> scratch_;
};

} // namespace blockamr::solvers
