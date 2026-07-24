// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Matrix-free Ginkgo CG solve of an AMReX MLLinOp system (single-level, CPU
// serial). The mat-vec is MLMG::apply, which computes out = L_inhom(in): the
// operator evaluated with the inhomogeneous BC data set via set_level_bc, so
// it is AFFINE, not linear. The solve therefore runs in residual-correction
// form: with x0 the incoming sol and c0 = L_inhom(0) the constant offset,
//   A_home(v) = sign * (L_inhom(v) - c0)          (linear)
//   A_home(delta) = sign * (rhs - L_inhom(x0)),   sol = x0 + delta.
// `sign` makes the operator SPD for CG: -1 for MLPoisson (L = +laplacian,
// negative-definite), +1 for MLABecLaplacian (alpha*a*phi - beta*div(b
// grad phi), already positive-definite). With homogeneous BCs and x0 = 0,
// c0 = 0 and r0 = rhs, so this reduces exactly to a plain CG solve of
// sign*L (the milestone-1 behavior).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <AMReX_Arena.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>

#include <ginkgo/ginkgo.hpp>

#include <nvtx3/nvToolsExt.h> // header-only NVTX v3 (M0 profiling ranges)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "bindings.hpp"

namespace nb = nanobind;

namespace
{

using MLMG = amrex::MLMGT<amrex::MultiFab>;
using Dense = gko::matrix::Dense<double>;

// ---------------------------------------------------------------------------
// M0 phase profiling. Env var BLOCKAMR_PROFILE (read once):
//   unset/0 : off — a single cached-int check per phase, no syncs, no NVTX.
//   1       : wall-clock phase timers, each phase bounded by
//             amrex::Gpu::streamSynchronize() on both ends (honest per-phase
//             attribution, but the extra syncs perturb the total), plus NVTX.
//   2       : NVTX ranges only, no extra syncs — for nsys GPU-projected
//             timelines of the unperturbed solve.
// Accumulated seconds/counts are exposed via profile_report()/profile_reset().
namespace prof
{

inline int mode()
{
    static const int m = []
    {
        const char* v = std::getenv("BLOCKAMR_PROFILE");
        return (v != nullptr && v[0] != '\0') ? std::atoi(v) : 0;
    }();
    return m;
}

struct Acc
{
    double sec = 0.0;
    long count = 0;
};

inline std::map<std::string, Acc>& table()
{
    static std::map<std::string, Acc> t;
    return t;
}

// Scoped phase timer; lvl >= 0 appends ".L<lvl>" (multigrid level) to the key.
class Timer
{
public:

    explicit Timer(const char* name, int lvl = -1)
    {
        if (mode() == 0)
        {
            return;
        }
        key_ = (lvl >= 0) ? std::string(name) + ".L" + std::to_string(lvl) : name;
        nvtxRangePushA(key_.c_str());
        if (mode() == 1)
        {
            amrex::Gpu::streamSynchronize();
            t0_ = std::chrono::steady_clock::now();
        }
    }

    ~Timer()
    {
        if (mode() == 0)
        {
            return;
        }
        if (mode() == 1)
        {
            amrex::Gpu::streamSynchronize();
            const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0_;
            auto& a = table()[key_];
            a.sec += dt.count();
            ++a.count;
        }
        nvtxRangePop();
    }

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

private:

    std::string key_;
    std::chrono::steady_clock::time_point t0_;
};

} // namespace prof

// Flat-vector <-> MultiFab transfer (component 0, valid cells only).
// gather and scatter MUST traverse cells in the identical order: MFIter
// without tiling, then k,j,i over the valid box. MultiFabs live in device
// memory by default in GPU builds, so access is staged through explicit
// host copies unless the arena is host-accessible. `scale` lets gather
// apply the SPD sign flip (-L) in the same pass.
// Templated on the FabArray type so the same host path serves the FP64
// MultiFab (Ginkgo double vector) and the FP32 GMG level fields
// (FabArray<BaseFab<float>>): the flat Ginkgo buffer is always double, so the
// per-cell read/write converts to/from the fab's value_type.
template<class FA>
void gather(const FA& mf, double* buf, double scale)
{
    using T = typename FA::value_type;
    const bool hostOk = mf.arena()->isHostAccessible();
    amrex::Gpu::streamSynchronize();
    std::size_t idx = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto& fab = mf[mfi];
        const amrex::Box& fbx = fab.box();
        std::vector<T> stage;
        auto arr = fab.const_array();
        if (!hostOk)
        {
            // Component 0 occupies the first numPts() elements of the fab.
            stage.resize(static_cast<std::size_t>(fbx.numPts()));
            amrex::Gpu::dtoh_memcpy(stage.data(), fab.dataPtr(), stage.size() * sizeof(T));
            arr = amrex::makeArray4<const T>(stage.data(), fbx, 1);
        }
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    buf[idx++] = scale * static_cast<double>(arr(i, j, k));
                }
            }
        }
    }
}

template<class FA>
void scatter(const double* buf, FA& mf)
{
    using T = typename FA::value_type;
    const bool hostOk = mf.arena()->isHostAccessible();
    amrex::Gpu::streamSynchronize();
    std::size_t idx = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        auto& fab = mf[mfi];
        const amrex::Box& fbx = fab.box();
        std::vector<T> stage;
        auto arr = fab.array();
        if (!hostOk)
        {
            // Round-trip the full fab so ghost values survive the update.
            stage.resize(static_cast<std::size_t>(fbx.numPts()));
            amrex::Gpu::dtoh_memcpy(stage.data(), fab.dataPtr(), stage.size() * sizeof(T));
            arr = amrex::makeArray4<T>(stage.data(), fbx, 1);
        }
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    arr(i, j, k) = static_cast<T>(buf[idx++]);
                }
            }
        }
        if (!hostOk)
        {
            amrex::Gpu::htod_memcpy(fab.dataPtr(), stage.data(), stage.size() * sizeof(T));
        }
    }
}

// Device pack/unpack between a contiguous Ginkgo vector (device memory) and a
// device-resident MultiFab, via amrex::ParallelFor so the whole mat-vec runs
// on the GPU with NO host round-trip per Krylov iteration. The flat index MUST
// match the host gather/scatter above (MFIter order; within a valid box the
// index runs fastest in i, then j, then k), because the one-time RHS pack and
// solution unpack in the solve still use the host path.
// Templated on the FabArray type (see the host twins): the flat Ginkgo vector
// is double; the fab may be double (FP64 path) or float (FP32 GMG level), so the
// per-cell copy converts through the fab's value_type on the device.
template<class FA>
void scatter_device(const double* vec, FA& mf)
{
    using T = typename FA::value_type;
    long off = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const int ni = vbx.length(0);
        const int nj = vbx.length(1);
        const long o = off;
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const long idx =
                    o + (static_cast<long>(k - lo.z) * nj + (j - lo.y)) * ni + (i - lo.x);
                a(i, j, k) = static_cast<T>(vec[idx]);
            }
        );
        off += vbx.numPts();
    }
}

template<class FA>
void gather_device(const FA& mf, double* vec, double scale)
{
    long off = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const int ni = vbx.length(0);
        const int nj = vbx.length(1);
        const long o = off;
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const long idx =
                    o + (static_cast<long>(k - lo.z) * nj + (j - lo.y)) * ni + (i - lo.x);
                vec[idx] = scale * static_cast<double>(a(i, j, k));
            }
        );
        off += vbx.numPts();
    }
}

// Matrix-free SPD operator: x = sign*(L_inhom(b) - c0), with MLMG::apply as
// the mat-vec and c0 = L_inhom(0) the affine BC offset recorded once at
// construction.
class AmrexOp : public gko::EnableLinOp<AmrexOp>, public gko::EnableCreateMethod<AmrexOp>
{
public:

    // Exec-only constructor required by the polymorphic-object machinery
    // (create_default / clear).
    explicit AmrexOp(std::shared_ptr<const gko::Executor> exec) : gko::EnableLinOp<AmrexOp>(exec) {}

    AmrexOp(
        std::shared_ptr<const gko::Executor> exec,
        MLMG* mlmg,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        gko::size_type n,
        double sign
    )
        : gko::EnableLinOp<AmrexOp>(exec, gko::dim<2> {n, n}), mlmg_(mlmg), sign_(sign),
          // shared_ptr, not values: MultiFab is move-only, but
          // EnablePolymorphicAssignment needs AmrexOp copy-assignable.
          // MLMG::apply needs >= 1 ghost cell on the input, hence ng=1 on in_.
          // Default (device) arena: on a Cuda executor the pack/unpack kernels
          // and MLMG::apply all run on the GPU with no host copies; on the
          // reference (CPU) path gather/scatter stage these via dtoh/htod.
          in_(std::make_shared<amrex::MultiFab>(ba, dm, 1, 1)),
          out_(std::make_shared<amrex::MultiFab>(ba, dm, 1, 0)),
          c0_(std::make_shared<amrex::MultiFab>(ba, dm, 1, 0))
    {
        in_->setVal(0.0);
        out_->setVal(0.0);
        // Affine offset c0 = L_inhom(0): captures the set_level_bc
        // contribution so apply_impl can subtract it and stay linear.
        mlmg_->apply({c0_.get()}, {in_.get()});
        // apply overwrites in_'s ghost cells; restore the all-zero state.
        in_->setVal(0.0);
        amrex::Gpu::streamSynchronize();
    }

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto exec = this->get_executor();
        // A ReferenceExecutor is its own master; a device (Cuda) executor has a
        // distinct host master. On device, pack/unpack run as AMReX kernels
        // straight against the Ginkgo device pointers, so the entire mat-vec
        // stays on the GPU. On host, stage through host clones.
        const bool onDevice = exec->get_master().get() != exec.get();
        if (onDevice)
        {
            // Ordering across the two libraries' streams is done host-side:
            // wait for Ginkgo's writes to b, run the AMReX mat-vec, then wait
            // for its writes to x before Ginkgo reads them.
            exec->synchronize();
            scatter_device(gko::as<Dense>(b)->get_const_values(), *in_);
            mlmg_->apply({out_.get()}, {in_.get()});
            amrex::MultiFab::Subtract(*out_, *c0_, 0, 0, 1, 0);
            gather_device(*out_, gko::as<Dense>(x)->get_values(), sign_);
            amrex::Gpu::streamSynchronize();
        }
        else
        {
            auto host = exec->get_master();
            auto bHost = gko::clone(host, gko::as<Dense>(b));
            scatter(bHost->get_const_values(), *in_);
            mlmg_->apply({out_.get()}, {in_.get()});
            // Remove the affine BC offset, then apply the SPD sign on gather:
            // x = sign*(L_inhom(in) - c0).
            amrex::MultiFab::Subtract(*out_, *c0_, 0, 0, 1, 0);
            auto xHost = Dense::create(host, gko::as<Dense>(x)->get_size());
            gather(*out_, xHost->get_values(), sign_);
            gko::as<Dense>(x)->copy_from(xHost);
        }
    }

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        auto denseX = gko::as<Dense>(x);
        auto tmp = denseX->clone();
        this->apply_impl(b, tmp.get());
        denseX->scale(beta);
        denseX->add_scaled(alpha, tmp);
    }

private:

    MLMG* mlmg_ = nullptr;
    double sign_ = -1.0;
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
    std::shared_ptr<amrex::MultiFab> c0_;
};

// Multi-level (composite AMR) generalisation of AmrexOp: the Ginkgo vector is
// the concatenation of all levels' valid cells (coarsest first, each level in
// the gather/scatter per-box flattening order) and the mat-vec is the
// multi-level MLMG::apply — the COMPOSITE operator: per level
// out[l] = L(in) with the fine level's coarse/fine boundary interpolated from
// the coarse `in`, the coarse residual refluxed at the coarse/fine interface
// (which cancels any dependence on coarse cells covered by the fine patch),
// and the covered coarse output overwritten by average_down of the fine
// output. Consequences for the linear system on the full concatenated vector:
//   - columns belonging to covered coarse cells are ZERO (index-1 singular;
//     nullspace = covered-cell perturbations, disjoint from the range), so a
//     consistent rhs (covered coarse rhs = average_down of the fine rhs —
//     enforced by the caller) is solvable and the covered solution entries
//     are fixed afterwards by a final average_down;
//   - the composite operator is NOT symmetric (the c/f ghost interpolation is
//     not the adjoint of the reflux), so bicgstab/gmres are the safe solvers
//     (CG may still work in practice — measured by the caller/tests).
// Affine offset c0 = L_inhom(0) recorded per level, as in AmrexOp.
class CompositeAmrexOp :
    public gko::EnableLinOp<CompositeAmrexOp>,
    public gko::EnableCreateMethod<CompositeAmrexOp>
{
public:

    explicit CompositeAmrexOp(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<CompositeAmrexOp>(exec)
    {}

    CompositeAmrexOp(
        std::shared_ptr<const gko::Executor> exec,
        MLMG* mlmg,
        const std::vector<amrex::BoxArray>& bas,
        const std::vector<amrex::DistributionMapping>& dms,
        gko::size_type n,
        double sign
    )
        : gko::EnableLinOp<CompositeAmrexOp>(exec, gko::dim<2> {n, n}), mlmg_(mlmg), sign_(sign)
    {
        long off = 0;
        for (std::size_t lev = 0; lev < bas.size(); ++lev)
        {
            // shared_ptr for copy-assignability (see AmrexOp); MLMG::apply
            // needs >= 1 ghost cell on the input, hence ng=1 on in_.
            in_.push_back(std::make_shared<amrex::MultiFab>(bas[lev], dms[lev], 1, 1));
            out_.push_back(std::make_shared<amrex::MultiFab>(bas[lev], dms[lev], 1, 0));
            c0_.push_back(std::make_shared<amrex::MultiFab>(bas[lev], dms[lev], 1, 0));
            in_.back()->setVal(0.0);
            out_.back()->setVal(0.0);
            off_.push_back(off);
            off += bas[lev].numPts();
        }
        // Affine offset c0 = L_inhom(0) per level (set_level_bc contribution).
        mlmg_->apply(ptrs(c0_), ptrs(in_));
        // apply overwrites in_'s ghost cells; restore the all-zero state.
        for (auto& mf : in_)
        {
            mf->setVal(0.0);
        }
        amrex::Gpu::streamSynchronize();
    }

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto exec = this->get_executor();
        const bool onDevice = exec->get_master().get() != exec.get();
        if (onDevice)
        {
            exec->synchronize(); // b written by Ginkgo
            const double* bv = gko::as<Dense>(b)->get_const_values();
            for (std::size_t lev = 0; lev < in_.size(); ++lev)
            {
                scatter_device(bv + off_[lev], *in_[lev]);
            }
            mlmg_->apply(ptrs(out_), ptrs(in_));
            double* xv = gko::as<Dense>(x)->get_values();
            for (std::size_t lev = 0; lev < out_.size(); ++lev)
            {
                amrex::MultiFab::Subtract(*out_[lev], *c0_[lev], 0, 0, 1, 0);
                gather_device(*out_[lev], xv + off_[lev], sign_);
            }
            amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
        }
        else
        {
            auto host = exec->get_master();
            auto bHost = gko::clone(host, gko::as<Dense>(b));
            for (std::size_t lev = 0; lev < in_.size(); ++lev)
            {
                scatter(bHost->get_const_values() + off_[lev], *in_[lev]);
            }
            mlmg_->apply(ptrs(out_), ptrs(in_));
            auto xHost = Dense::create(host, gko::as<Dense>(x)->get_size());
            for (std::size_t lev = 0; lev < out_.size(); ++lev)
            {
                amrex::MultiFab::Subtract(*out_[lev], *c0_[lev], 0, 0, 1, 0);
                gather(*out_[lev], xHost->get_values() + off_[lev], sign_);
            }
            gko::as<Dense>(x)->copy_from(xHost);
        }
    }

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        auto denseX = gko::as<Dense>(x);
        auto tmp = denseX->clone();
        this->apply_impl(b, tmp.get());
        denseX->scale(beta);
        denseX->add_scaled(alpha, tmp);
    }

private:

    static amrex::Vector<amrex::MultiFab*> ptrs(
        const std::vector<std::shared_ptr<amrex::MultiFab>>& v
    )
    {
        amrex::Vector<amrex::MultiFab*> p;
        for (const auto& m : v)
        {
            p.push_back(m.get());
        }
        return p;
    }

    MLMG* mlmg_ = nullptr;
    double sign_ = 1.0;
    std::vector<std::shared_ptr<amrex::MultiFab>> in_;
    std::vector<std::shared_ptr<amrex::MultiFab>> out_;
    std::vector<std::shared_ptr<amrex::MultiFab>> c0_;
    std::vector<long> off_;
};

// Multigrid preconditioner: z = M^{-1} r approximated by a FIXED small number
// of MLMG V-cycles (setFixedIter) on a caller-supplied equivalent operator.
// Used as the generated preconditioner of the matrix-free Krylov solve, so the
// iteration count stays ~flat in N (MG) while the outer mat-vec stays
// matrix-free. The loose tolerances passed to solve() are ignored in
// fixed-iter mode. NOTE: a V-cycle with (red-black) Gauss-Seidel smoothing is
// only approximately symmetric — classic CG tolerates it here (measured), but
// bicgstab/gmres are the fallback if it ever degrades.
class MlmgPrecond :
    public gko::EnableLinOp<MlmgPrecond>,
    public gko::EnableCreateMethod<MlmgPrecond>
{
public:

    explicit MlmgPrecond(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<MlmgPrecond>(exec)
    {}

    MlmgPrecond(
        std::shared_ptr<const gko::Executor> exec,
        MLMG* mlmg,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        gko::size_type n,
        int n_cycles
    )
        : gko::EnableLinOp<MlmgPrecond>(exec, gko::dim<2> {n, n}), mlmg_(mlmg),
          // shared_ptr for copy-assignability (see AmrexOp). Default (device)
          // arena: on a Cuda executor pack/unpack and the V-cycles all run on
          // the GPU; the reference path stages via scatter/gather.
          in_(std::make_shared<amrex::MultiFab>(ba, dm, 1, 1)),
          out_(std::make_shared<amrex::MultiFab>(ba, dm, 1, 1))
    {
        in_->setVal(0.0);
        out_->setVal(0.0);
        mlmg_->setVerbose(0);
        mlmg_->setBottomVerbose(0);
        // Exactly n_cycles V-cycles per apply; solve() tolerances are ignored.
        mlmg_->setFixedIter(n_cycles);
        amrex::Gpu::streamSynchronize();
    }

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto exec = this->get_executor();
        const bool onDevice = exec->get_master().get() != exec.get();
        if (onDevice)
        {
            exec->synchronize(); // b written by Ginkgo
            scatter_device(gko::as<Dense>(b)->get_const_values(), *in_);
            out_->setVal(0.0); // z0 = 0: apply M^{-1}, not a warm-started solve
            mlmg_->solve({out_.get()}, {in_.get()}, 1e-4, 0.0);
            gather_device(*out_, gko::as<Dense>(x)->get_values(), 1.0);
            amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
        }
        else
        {
            auto host = exec->get_master();
            auto bHost = gko::clone(host, gko::as<Dense>(b));
            scatter(bHost->get_const_values(), *in_);
            out_->setVal(0.0);
            mlmg_->solve({out_.get()}, {in_.get()}, 1e-4, 0.0);
            auto xHost = Dense::create(host, gko::as<Dense>(x)->get_size());
            gather(*out_, xHost->get_values(), 1.0);
            gko::as<Dense>(x)->copy_from(xHost);
        }
    }

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        auto denseX = gko::as<Dense>(x);
        auto tmp = denseX->clone();
        this->apply_impl(b, tmp.get());
        denseX->scale(beta);
        denseX->add_scaled(alpha, tmp);
    }

private:

    MLMG* mlmg_ = nullptr;
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

// Host-accessible (pinned) copy of a MultiFab. The coefficient fields arrive
// in the default arena — device memory in a GPU build — but the face-coeff stencil
// runs host-side on the ReferenceExecutor, so the (solve-constant) coefficients
// are staged to pinned memory once at operator construction.
std::shared_ptr<amrex::MultiFab> pinnedCopy(const amrex::MultiFab& src)
{
    auto dst = std::make_shared<amrex::MultiFab>(
        src.boxArray(),
        src.DistributionMap(),
        src.nComp(),
        src.nGrow(),
        amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
    );
    amrex::MultiFab::Copy(*dst, src, 0, 0, src.nComp(), src.nGrow());
    amrex::Gpu::streamSynchronize();
    return dst;
}

// Domain-boundary condition per side, order (xlo, xhi, ylo, yhi, zlo, zhi):
// 0 = periodic (handled by FillBoundary), 1 = homogeneous Dirichlet (u = 0 on
// the face), 2 = homogeneous Neumann (du/dn = 0 on the face).
using BcArray = std::array<int, 6>;

BcArray parseBc(
    const std::vector<std::string>& bc, const amrex::Geometry& geom, const std::string& who
)
{
    if (bc.size() != 6)
    {
        throw std::runtime_error(who + ": bc must have 6 entries (xlo, xhi, ylo, yhi, zlo, zhi)");
    }
    BcArray out {};
    for (int s = 0; s < 6; ++s)
    {
        const std::string& v = bc[static_cast<std::size_t>(s)];
        if (v == "periodic")
        {
            out[static_cast<std::size_t>(s)] = 0;
        }
        else if (v == "dirichlet")
        {
            out[static_cast<std::size_t>(s)] = 1;
        }
        else if (v == "neumann")
        {
            out[static_cast<std::size_t>(s)] = 2;
        }
        else
        {
            throw std::runtime_error(
                who + ": unknown bc '" + v + "' (expected 'periodic', 'dirichlet' or 'neumann')"
            );
        }
        const int dim = s / 2;
        if (geom.isPeriodic(dim) && v != "periodic")
        {
            throw std::runtime_error(
                who + ": bc '" + v + "' on periodic geometry direction " + std::to_string(dim)
                + " — make the direction non-periodic or use bc='periodic'"
            );
        }
        if (!geom.isPeriodic(dim) && v == "periodic")
        {
            throw std::runtime_error(
                who + ": bc 'periodic' on non-periodic geometry direction " + std::to_string(dim)
            );
        }
    }
    return out;
}

// Ghost-layer fill spec for domain side s (0..5) of a valid box: the
// one-cell-thick ghost layer to write, the reflection sign (-1 Dirichlet
// reflect-odd, +1 Neumann reflect-even) and the offset from each ghost cell to
// its mirror interior cell. Returns false when the side is periodic or the box
// does not touch that domain face.
struct BcGhostFill
{
    amrex::Box gbx;
    double sign;
    int di, dj, dk;
};

bool bcGhostFill(
    const amrex::Box& vbx, const amrex::Box& domain, const BcArray& bc, int s, BcGhostFill& f
)
{
    if (bc[static_cast<std::size_t>(s)] == 0)
    {
        return false;
    }
    const int dir = s / 2;
    const bool low = (s % 2) == 0;
    const bool touches = low ? vbx.smallEnd(dir) == domain.smallEnd(dir)
                             : vbx.bigEnd(dir) == domain.bigEnd(dir);
    if (!touches)
    {
        return false;
    }
    const int gpos = low ? vbx.smallEnd(dir) - 1 : vbx.bigEnd(dir) + 1;
    f.gbx = vbx;
    f.gbx.setSmall(dir, gpos);
    f.gbx.setBig(dir, gpos);
    f.sign = (bc[static_cast<std::size_t>(s)] == 1) ? -1.0 : 1.0;
    const int shift = low ? 1 : -1;
    f.di = (dir == 0) ? shift : 0;
    f.dj = (dir == 1) ? shift : 0;
    f.dk = (dir == 2) ? shift : 0;
    return true;
}

// Fill the domain-boundary ghost layer of `mf` (1 ghost, component 0) so the
// face-coefficient stencil folds homogeneous BCs with the matrix untouched:
// Dirichlet -> ghost = -interior (u = 0 at the face, 2nd order at the dx/2
// face distance), Neumann -> ghost = interior (du/dn = 0). Only face ghost
// layers are needed — the 7-point stencil never reads edge/corner ghosts.
// Free function: nvcc forbids an extended __device__ lambda inside a
// protected/private member.
// Templated on the FabArray type: serves the FP64 operator MultiFab and the
// FP32 GMG level fabs; `value_type` sizes the sign/reflection cast.
template<class FA>
void fillDomainBcGhostsDevice(FA& mf, const amrex::Box& domain, const BcArray& bc)
{
    using T = typename FA::value_type;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.array(mfi);
        for (int s = 0; s < 6; ++s)
        {
            BcGhostFill f;
            if (!bcGhostFill(vbx, domain, bc, s, f))
            {
                continue;
            }
            const T sign = static_cast<T>(f.sign);
            const int di = f.di, dj = f.dj, dk = f.dk;
            amrex::ParallelFor(
                f.gbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { a(i, j, k) = sign * a(i + di, j + dj, k + dk); }
            );
        }
    }
}

// Host-loop twin of fillDomainBcGhostsDevice for the ReferenceExecutor path.
template<class FA>
void fillDomainBcGhostsHost(FA& mf, const amrex::Box& domain, const BcArray& bc)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.array(mfi);
        for (int s = 0; s < 6; ++s)
        {
            BcGhostFill f;
            if (!bcGhostFill(vbx, domain, bc, s, f))
            {
                continue;
            }
            const auto lo = amrex::lbound(f.gbx);
            const auto hi = amrex::ubound(f.gbx);
            for (int k = lo.z; k <= hi.z; ++k)
            {
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        a(i, j, k) = f.sign * a(i + f.di, j + f.dj, k + f.dk);
                    }
                }
            }
        }
    }
}

// Scatter ONLY the ghost-adjacent shell (outer 1-cell layer of each valid box)
// from the flat Ginkgo vector into the MultiFab (M3 3a). That shell is all that
// FillBoundary (periodic/internal) and the reflect domain-BC fill read to
// populate the face ghosts the fused stencil consults; the interior valid cells
// are read straight from the flat vector by faceCoeffStencilFusedDevice, so they
// need not be copied. Flat index matches scatter_device (box-by-box, i fastest).
void scatterShellDevice(const double* vec, amrex::MultiFab& mf)
{
    long off = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        const int ni = vbx.length(0);
        const int nj = vbx.length(1);
        const long o = off;
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if (i == lo.x || i == hi.x || j == lo.y || j == hi.y || k == lo.z || k == hi.z)
                {
                    const long idx =
                        o + (static_cast<long>(k - lo.z) * nj + (j - lo.y)) * ni + (i - lo.x);
                    a(i, j, k) = vec[idx];
                }
            }
        );
        off += vbx.numPts();
    }
}

// Fused matrix-free apply (M3 3a) that skips the full flat<->MultiFab pack/unpack:
// the stencil reads the centre and any interior neighbour straight from the flat
// Ginkgo input `bvec`, consulting the ghosted scratch `in` ONLY for a neighbour
// that leaves the valid box (periodic/internal/domain-BC ghost, filled from the
// shell scatter + FillBoundary), and writes the result straight into the flat
// output `xvec`. No out_ MultiFab, no gather. Bit-identical to the plain
// face-coefficient stencil + full scatter/gather (interior flat values equal the
// scattered in_ values). Flat index matches scatter_device. Assumes b/x do not
// alias (Krylov apply never aliases operand and result).
void faceCoeffStencilFusedDevice(
    const double* bvec,
    double* xvec,
    const amrex::MultiFab& in,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alpha
)
{
    long off = 0;
    for (amrex::MFIter mfi(in); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = in.const_array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        const int ni = vbx.length(0);
        const int nj = vbx.length(1);
        const long nij = static_cast<long>(ni) * nj;
        const long o = off;
        const double* b = bvec;
        double* xo = xvec;
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const long idx =
                    o + (static_cast<long>(k - lo.z) * nj + (j - lo.y)) * ni + (i - lo.x);
                const double pC = b[idx];
                const double pE = (i < hi.x) ? b[idx + 1] : psi(i + 1, j, k);
                const double pW = (i > lo.x) ? b[idx - 1] : psi(i - 1, j, k);
                const double pN = (j < hi.y) ? b[idx + ni] : psi(i, j + 1, k);
                const double pS = (j > lo.y) ? b[idx - ni] : psi(i, j - 1, k);
                const double pT = (k < hi.z) ? b[idx + nij] : psi(i, j, k + 1);
                const double pB = (k > lo.z) ? b[idx - nij] : psi(i, j, k - 1);
                const double aE = ax(i + 1, j, k);
                const double aW = lxa(i, j, k);
                const double aN = ay(i, j + 1, k);
                const double aS = lya(i, j, k);
                const double aT = az(i, j, k + 1);
                const double aB = lza(i, j, k);
                const double offd = aE * pE + aW * pW + aN * pN + aS * pS + aT * pT + aB * pB;
                const double diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                xo[idx] = diag * pC + offd;
            }
        );
        off += vbx.numPts();
    }
}

// General matrix-free face-coefficient operator on a structured single-level grid. The
// matrix is carried as OpenFOAM-style pieces given as AMReX fields:
//   alpha  : cell-centred diagonal SOURCE (ddt/Sp/reaction), NOT the full
//            diagonal — the face part is derived below (negSumDiag).
//   u{x,y,z}, l{x,y,z} : face-centred upper/lower off-diagonal coefficients.
//             u* is the owner-row->neighbour coupling on the cell's HIGH face,
//            l* the neighbour-row->owner coupling on the cell's LOW face. For a
//            symmetric matrix pass the same MultiFab for u* and l*.
// The mat-vec is the OpenFOAM Amul in pull form (each cell reads its 6
// neighbours), with the diagonal assembled on the fly as
//   diag = alpha - (aE+aW+aN+aS+aT+aB)               (negSumDiag)
// so no cell-diagonal array is stored — the face coeffs feed both the
// off-diagonal and the diagonal. This is exact whenever the flux part
// annihilates a constant (divergence-free flux / pure diffusion); any
// non-conservative diagonal contribution must be folded into alpha.
class FaceCoeffOp :
    public gko::EnableLinOp<FaceCoeffOp>,
    public gko::EnableCreateMethod<FaceCoeffOp>
{
public:

    explicit FaceCoeffOp(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<FaceCoeffOp>(exec)
    {}

    FaceCoeffOp(
        std::shared_ptr<const gko::Executor> exec,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        amrex::Geometry geom,
        gko::size_type n,
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        BcArray bc = {}
    )
        : gko::EnableLinOp<FaceCoeffOp>(exec, gko::dim<2> {n, n}), geom_(geom), bc_(bc),
          hasPhysBc_(std::any_of(bc.begin(), bc.end(), [](int b) { return b != 0; })),
          onDevice_(exec->get_master().get() != exec.get())
    {
        if (onDevice_)
        {
            // Reference the caller's device fields directly; the stencil reads
            // them on the GPU and in_/out_ live in the default (device) arena.
            alpha_ = alpha;
            ux_ = ux;
            lx_ = lx;
            uy_ = uy;
            ly_ = ly;
            uz_ = uz;
            lz_ = lz;
            in_ = std::make_shared<amrex::MultiFab>(ba, dm, 1, 1);
            out_ = std::make_shared<amrex::MultiFab>(ba, dm, 1, 0);
        }
        else
        {
            // Host (ReferenceExecutor) stencil: stage the coefficients to
            // pinned memory once and read those.
            owned_ = {
                pinnedCopy(*alpha),
                pinnedCopy(*ux),
                pinnedCopy(*lx),
                pinnedCopy(*uy),
                pinnedCopy(*ly),
                pinnedCopy(*uz),
                pinnedCopy(*lz)
            };
            alpha_ = owned_[0].get();
            ux_ = owned_[1].get();
            lx_ = owned_[2].get();
            uy_ = owned_[3].get();
            ly_ = owned_[4].get();
            uz_ = owned_[5].get();
            lz_ = owned_[6].get();
            in_ = std::make_shared<amrex::MultiFab>(
                ba, dm, 1, 1, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
            );
            out_ = std::make_shared<amrex::MultiFab>(
                ba, dm, 1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
            );
        }
        in_->setVal(0.0);
        out_->setVal(0.0);
    }

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        if (onDevice_)
        {
            prof::Timer tAll("op.apply");
            {
                prof::Timer t("op.sync_gko");
                this->get_executor()->synchronize(); // b written by Ginkgo
            }
            const double* bvals = gko::as<Dense>(b)->get_const_values();
            double* xvals = gko::as<Dense>(x)->get_values();
            {
                // M3 3a: only the ghost-adjacent shell needs to reach the MF —
                // FillBoundary/domain-BC read it to fill the face ghosts; the
                // interior is read straight from the flat vector by the stencil.
                prof::Timer t("op.scatter");
                scatterShellDevice(bvals, *in_);
            }
            {
                prof::Timer t("op.fill");
                in_->FillBoundary(geom_.periodicity());
                if (hasPhysBc_)
                {
                    // Domain-boundary ghosts: reflect-odd/even folds the
                    // homogeneous Dirichlet/Neumann BCs into the stencil.
                    fillDomainBcGhostsDevice(*in_, geom_.Domain(), bc_);
                }
            }
            amrex::Gpu::streamSynchronize();
            {
                prof::Timer t("op.stencil");
                // Fused: reads interior neighbours from the flat vector, ghosts
                // from in_, writes straight to the flat output (no gather). Free
                // function: nvcc forbids an extended __device__ lambda in a member.
                faceCoeffStencilFusedDevice(
                    bvals, xvals, *in_, *ux_, *lx_, *uy_, *ly_, *uz_, *lz_, *alpha_
                );
            }
            {
                prof::Timer t("op.gather");
                amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
            }
            return;
        }

        scatter(gko::as<Dense>(b)->get_const_values(), *in_);
        // Fill periodic + internal-box ghosts. Physical-boundary ghosts are
        // then set by the reflect fill below when bc has dirichlet/neumann
        // sides; on all-periodic operators they stay whatever scatter left
        // (untouched valid-only write) and the boundary faces must carry a
        // zero coefficient for those to be harmless.
        in_->FillBoundary(geom_.periodicity());
        amrex::Gpu::streamSynchronize();
        if (hasPhysBc_)
        {
            fillDomainBcGhostsHost(*in_, geom_.Domain(), bc_);
        }

        for (amrex::MFIter mfi(*out_); mfi.isValid(); ++mfi)
        {
            const amrex::Box& vbx = mfi.validbox();
            const auto psi = in_->const_array(mfi);
            const auto o = out_->array(mfi);
            const auto ax = ux_->const_array(mfi);
            const auto lxa = lx_->const_array(mfi);
            const auto ay = uy_->const_array(mfi);
            const auto lya = ly_->const_array(mfi);
            const auto az = uz_->const_array(mfi);
            const auto lza = lz_->const_array(mfi);
            const auto al = alpha_->const_array(mfi);
            const auto lo = amrex::lbound(vbx);
            const auto hi = amrex::ubound(vbx);
            for (int k = lo.z; k <= hi.z; ++k)
            {
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        // Off-diagonals: aE=ux(high face), aW=lx(low face), etc.
                        const double aE = ax(i + 1, j, k);
                        const double aW = lxa(i, j, k);
                        const double aN = ay(i, j + 1, k);
                        const double aS = lya(i, j, k);
                        const double aT = az(i, j, k + 1);
                        const double aB = lza(i, j, k);
                        const double off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                         + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                         + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                        const double diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                        o(i, j, k) = diag * psi(i, j, k) + off;
                    }
                }
            }
        }
        gather(*out_, gko::as<Dense>(x)->get_values(), 1.0);
    }

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        auto denseX = gko::as<Dense>(x);
        auto tmp = denseX->clone();
        this->apply_impl(b, tmp.get());
        denseX->scale(beta);
        denseX->add_scaled(alpha, tmp);
    }

private:

    amrex::Geometry geom_;
    BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool onDevice_ = false;
    // Host path: owns pinned copies of the coefficient fields. Device path:
    // empty, and the pointers below reference the caller's device-resident
    // fields directly, so an external in-place update to the coefficients is
    // picked up by the next apply with no reassembly.
    std::vector<std::shared_ptr<amrex::MultiFab>> owned_;
    const amrex::MultiFab* alpha_ = nullptr;
    const amrex::MultiFab* ux_ = nullptr;
    const amrex::MultiFab* lx_ = nullptr;
    const amrex::MultiFab* uy_ = nullptr;
    const amrex::MultiFab* ly_ = nullptr;
    const amrex::MultiFab* uz_ = nullptr;
    const amrex::MultiFab* lz_ = nullptr;
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

// ---------------------------------------------------------------------------
// Native geometric-multigrid V-cycle preconditioner (GmgPrecond) kernels.
// Built from AMReX primitives on the face-coefficient operator only — no
// MLLinOp/MLMG anywhere in this path. Device kernels are namespace-scope free
// functions (nvcc: no extended __device__ lambdas in private/protected
// members) with host-loop twins for the ReferenceExecutor path.
//
// Every kernel is templated on the level value type T (double for the default
// FP64 hierarchy, float for the M5 gmg_precision="fp32" hierarchy): the whole
// V-cycle — level coefficients, sol/rhs work fields, smoother, residual /
// restriction / prolongation, ghost fills and the λmax power iteration — runs in
// T while the outer CG/operator stays FP64. GmgFab<T> is the level fab type.
// ---------------------------------------------------------------------------

template<class T>
using GmgFab = amrex::FabArray<amrex::BaseFab<T>>;

// Tiny |diagonal| floor guarding the RB-GS in-place division (skip rather than
// divide by ~0). Per value type so the double path keeps its 1e-300 floor
// exactly while the float path uses a representable one (1e-300 is not a valid
// float literal).
template<class T>
AMREX_GPU_HOST_DEVICE constexpr T gmgDiagFloor();
template<>
AMREX_GPU_HOST_DEVICE constexpr double gmgDiagFloor<double>()
{
    return 1e-300;
}
template<>
AMREX_GPU_HOST_DEVICE constexpr float gmgDiagFloor<float>()
{
    return 1e-30f;
}

// Copy src (any FabArray, e.g. the caller's FP64 MultiFab or a same-type level
// fab) into the T-valued dst, converting per cell over dst's valid box. Replaces
// MultiFab::Copy on the FP32 path (which requires matching value types); for
// T=double it is an exact copy, so the FP64 path is numerically unchanged.
template<class T, class SRC>
void gmgConvertCopyDevice(GmgFab<T>& dst, const SRC& src)
{
    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto d = dst.array(mfi);
        const auto s = src.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { d(i, j, k) = static_cast<T>(s(i, j, k)); }
        );
    }
}

template<class T, class SRC>
void gmgConvertCopyHost(GmgFab<T>& dst, const SRC& src)
{
    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto d = dst.array(mfi);
        const auto s = src.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    d(i, j, k) = static_cast<T>(s(i, j, k));
                }
            }
        }
    }
}

// dst += src, per cell over dst's valid box, converting through dst's value_type.
// dst is any FabArray (the caller's FP64 MultiFab); src is a T-valued level fab.
// The native stationary solver adds the (possibly FP32) V-cycle correction back
// onto the FP64 solution; for both double it is a plain in-place add.
template<class DST, class T>
void gmgConvertAddDevice(DST& dst, const GmgFab<T>& src)
{
    using DT = typename DST::value_type;
    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto d = dst.array(mfi);
        const auto s = src.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { d(i, j, k) += static_cast<DT>(s(i, j, k)); }
        );
    }
}

template<class DST, class T>
void gmgConvertAddHost(DST& dst, const GmgFab<T>& src)
{
    using DT = typename DST::value_type;
    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto d = dst.array(mfi);
        const auto s = src.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    d(i, j, k) += static_cast<DT>(s(i, j, k));
                }
            }
        }
    }
}

// Fused residual + convert-scatter + norm for the native GMG stationary solver
// (M3 target 3). Computes r = rhs - A*sol - shift in DOUBLE (shift is the
// nullspace-projection constant, 0 when not projecting) and stores it (cast to T)
// straight into the L0 rhs fab `out` — no separate FP64 residual MultiFab and no
// convert-scatter pass (M3 3a). The norm is a SECOND, light kernel reducing `out`.
//
// Why two kernels, not one: folding the sum(r^2) reduction INTO the heavy stencil
// kernel (10 coefficient/field Array4 + double arithmetic) was measured to cost
// ~1.0 ms/iter at 256^3 — the reduction machinery spills the register-bound
// kernel and slows the whole pass, exceeding the 0.34+0.54 ms it saves. A separate
// reduction over the freshly-written `out` is only ~0.20 ms/iter (light kernel,
// stays at the bandwidth roofline) and reuses the just-cached data.
//
// Precision of the norm: in the DEFAULT fp64 hierarchy (T=double) `out` holds the
// exact double residual, so the reduced norm is bit-exact FP64 — the convergence
// authority is unchanged. In the fp32 hierarchy (T=float) `out` holds the residual
// rounded to float; the reduced norm therefore carries ~6e-8 relative rounding,
// far below the ~10x per-cycle residual drop, so the stopping cycle is unchanged
// (verified: iters and converged answer identical to the FP64-norm path). Returns
// the FP64 sum of squares (caller takes the sqrt). Device + host twins.
template<class T>
double faceCoeffResidScatterNormDevice(
    const amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alpha,
    double shift,
    GmgFab<T>& out
)
{
    double res;
    {
        prof::Timer t("gmg.solve.residkern");
        for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
        {
            const amrex::Box& vbx = mfi.validbox();
            const auto psi = sol.const_array(mfi);
            const auto bb = rhs.const_array(mfi);
            const auto o = out.array(mfi);
            const auto ax = ux.const_array(mfi);
            const auto lxa = lx.const_array(mfi);
            const auto ay = uy.const_array(mfi);
            const auto lya = ly.const_array(mfi);
            const auto az = uz.const_array(mfi);
            const auto lza = lz.const_array(mfi);
            const auto al = alpha.const_array(mfi);
            amrex::ParallelFor(
                vbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const double aE = ax(i + 1, j, k);
                    const double aW = lxa(i, j, k);
                    const double aN = ay(i, j + 1, k);
                    const double aS = lya(i, j, k);
                    const double aT = az(i, j, k + 1);
                    const double aB = lza(i, j, k);
                    const double offd = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                      + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                      + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const double diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    const double r = bb(i, j, k) - (diag * psi(i, j, k) + offd) - shift;
                    o(i, j, k) = static_cast<T>(r);
                }
            );
        }
    }
    {
        prof::Timer t("gmg.solve.normkern");
        const auto o_ma = out.const_arrays();
        res = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpSum> {},
            amrex::TypeList<double> {},
            out,
            amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) -> amrex::GpuTuple<double>
            {
                const double v = static_cast<double>(o_ma[box](i, j, k));
                return {v * v};
            }
        );
    }
    return res;
}

template<class T>
double faceCoeffResidScatterNormHost(
    const amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alpha,
    double shift,
    GmgFab<T>& out
)
{
    double sumsq = 0.0;
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto bb = rhs.const_array(mfi);
        const auto o = out.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const double aE = ax(i + 1, j, k);
                    const double aW = lxa(i, j, k);
                    const double aN = ay(i, j + 1, k);
                    const double aS = lya(i, j, k);
                    const double aT = az(i, j, k + 1);
                    const double aB = lza(i, j, k);
                    const double offd = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                      + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                      + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const double diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    const double r = bb(i, j, k) - (diag * psi(i, j, k) + offd) - shift;
                    o(i, j, k) = static_cast<T>(r);
                    // Reduce the STORED value (like the device twin's separate
                    // ParReduce over `out`) so reference and cuda give an identical
                    // norm: exact FP64 for T=double, fp32-rounded for T=float.
                    const double v = static_cast<double>(o(i, j, k));
                    sumsq += v * v;
                }
            }
        }
    }
    return sumsq;
}

// ||mf||_2 over the valid region (0 ghost), accumulated in the fab's value_type
// (single-box/single-rank hierarchy; used only by the setup power iteration).
template<class T>
double gmgNorm2(const GmgFab<T>& mf)
{
    const T sq = amrex::ReduceSum(
        mf,
        amrex::IntVect(0),
        [=] AMREX_GPU_HOST_DEVICE(
            const amrex::Box& bx, const amrex::Array4<const T>& a
        ) -> T
        {
            T s = 0;
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            for (int k = lo.z; k <= hi.z; ++k)
            {
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        s += a(i, j, k) * a(i, j, k);
                    }
                }
            }
            return s;
        }
    );
    return std::sqrt(static_cast<double>(sq));
}

// One red-black Gauss-Seidel colour pass: cells with (i+j+k) parity `parity`
// are solved exactly in place, sol = (rhs - off) / D with D = alpha -
// sum(face coeffs) recomputed on the fly (tiny |D| guarded to no update). The
// 7-point stencil only couples opposite colours, so the in-place update is
// race-free. sol's ghosts must be refreshed before EACH colour pass.
template<class T>
void gmgGsColorDevice(
    GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha,
    int parity
)
{
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if (((i + j + k) & 1) != parity)
                {
                    return;
                }
                const T aE = ax(i + 1, j, k);
                const T aW = lxa(i, j, k);
                const T aN = ay(i, j + 1, k);
                const T aS = lya(i, j, k);
                const T aT = az(i, j, k + 1);
                const T aB = lza(i, j, k);
                const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                            + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                            + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                if (amrex::Math::abs(diag) > gmgDiagFloor<T>())
                {
                    psi(i, j, k) = (b(i, j, k) - off) / diag;
                }
            }
        );
    }
}

template<class T>
void gmgGsColorHost(
    GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha,
    int parity
)
{
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    if (((i + j + k) & 1) != parity)
                    {
                        continue;
                    }
                    const T aE = ax(i + 1, j, k);
                    const T aW = lxa(i, j, k);
                    const T aN = ay(i, j + 1, k);
                    const T aS = lya(i, j, k);
                    const T aT = az(i, j, k + 1);
                    const T aB = lza(i, j, k);
                    const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    if (std::abs(diag) > gmgDiagFloor<T>())
                    {
                        psi(i, j, k) = (b(i, j, k) - off) / diag;
                    }
                }
            }
        }
    }
}

// Volume-average (factor-2) restriction of a cell field: coarse = mean of the
// 8 fine children. Also used to coarsen alpha (a per-volume density). Iterates
// the coarse MF; the fine MF shares the DistributionMapping, so the same MFIter
// index addresses the matching fine box (its BoxArray is refine(coarse, 2)).
template<class T>
void gmgRestrictDevice(const GmgFab<T>& fine, GmgFab<T>& crse)
{
    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto f = fine.const_array(mfi);
        const auto c = crse.array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const int i2 = 2 * i, j2 = 2 * j, k2 = 2 * k;
                c(i, j, k) = static_cast<T>(0.125)
                           * (f(i2, j2, k2) + f(i2 + 1, j2, k2) + f(i2, j2 + 1, k2)
                              + f(i2 + 1, j2 + 1, k2) + f(i2, j2, k2 + 1) + f(i2 + 1, j2, k2 + 1)
                              + f(i2, j2 + 1, k2 + 1) + f(i2 + 1, j2 + 1, k2 + 1));
            }
        );
    }
}

template<class T>
void gmgRestrictHost(const GmgFab<T>& fine, GmgFab<T>& crse)
{
    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto f = fine.const_array(mfi);
        const auto c = crse.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const int i2 = 2 * i, j2 = 2 * j, k2 = 2 * k;
                    c(i, j, k) = static_cast<T>(0.125)
                               * (f(i2, j2, k2) + f(i2 + 1, j2, k2) + f(i2, j2 + 1, k2)
                                  + f(i2 + 1, j2 + 1, k2) + f(i2, j2, k2 + 1)
                                  + f(i2 + 1, j2, k2 + 1) + f(i2, j2 + 1, k2 + 1)
                                  + f(i2 + 1, j2 + 1, k2 + 1));
                }
            }
        }
    }
}

// Coarsen a face-coefficient field in direction `dir`: coarse face i_c covers
// fine face 2*i_c with the 2x2 transverse fine faces; a ~ -beta/dx^2, so the
// coarse coefficient is the arithmetic average of those 4 fine coefficients
// (beta averaged) divided by `scale` (dx doubled -> 4 for rediscretisation).
template<class T>
void gmgCoarsenFaceDevice(const GmgFab<T>& fine, GmgFab<T>& crse, int dir, double scale)
{
    int u[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    // The two transverse (cell) directions of face-normal `dir`.
    if (dir == 0) { u[1] = 1; v[2] = 1; }
    else if (dir == 1) { u[0] = 1; v[2] = 1; }
    else { u[0] = 1; v[1] = 1; }
    const int u0 = u[0], u1 = u[1], u2 = u[2];
    const int v0 = v[0], v1 = v[1], v2 = v[2];
    const T w = static_cast<T>(0.25 / scale);
    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto f = fine.const_array(mfi);
        const auto c = crse.array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const int i2 = 2 * i, j2 = 2 * j, k2 = 2 * k;
                c(i, j, k) = w
                           * (f(i2, j2, k2) + f(i2 + u0, j2 + u1, k2 + u2)
                              + f(i2 + v0, j2 + v1, k2 + v2)
                              + f(i2 + u0 + v0, j2 + u1 + v1, k2 + u2 + v2));
            }
        );
    }
}

template<class T>
void gmgCoarsenFaceHost(const GmgFab<T>& fine, GmgFab<T>& crse, int dir, double scale)
{
    int u[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    if (dir == 0) { u[1] = 1; v[2] = 1; }
    else if (dir == 1) { u[0] = 1; v[2] = 1; }
    else { u[0] = 1; v[1] = 1; }
    const T w = static_cast<T>(0.25 / scale);
    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto f = fine.const_array(mfi);
        const auto c = crse.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const int i2 = 2 * i, j2 = 2 * j, k2 = 2 * k;
                    c(i, j, k) = w
                               * (f(i2, j2, k2) + f(i2 + u[0], j2 + u[1], k2 + u[2])
                                  + f(i2 + v[0], j2 + v[1], k2 + v[2])
                                  + f(i2 + u[0] + v[0], j2 + u[1] + v[1], k2 + u[2] + v[2]));
                }
            }
        }
    }
}

// Piecewise-constant prolongation + correction: fine cell += coarse parent
// value (the adjoint of the volume-average restriction, up to the 1/8 factor).
template<class T>
void gmgProlongAddDevice(const GmgFab<T>& crse, GmgFab<T>& fine)
{
    for (amrex::MFIter mfi(fine); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto c = crse.const_array(mfi);
        const auto f = fine.array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { f(i, j, k) += c(amrex::coarsen(i, 2), amrex::coarsen(j, 2), amrex::coarsen(k, 2)); }
        );
    }
}

template<class T>
void gmgProlongAddHost(const GmgFab<T>& crse, GmgFab<T>& fine)
{
    for (amrex::MFIter mfi(fine); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto c = crse.const_array(mfi);
        const auto f = fine.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    f(i, j, k) +=
                        c(amrex::coarsen(i, 2), amrex::coarsen(j, 2), amrex::coarsen(k, 2));
                }
            }
        }
    }
}

// Fused residual + volume-average restriction: coarse rhs cell = mean of the 8
// fine residuals r = rhs - A sol, each computed on the fly. Iterates the coarse
// box (fine sol's ghosts must be filled). Saves the full fine-grid resid
// read+write of the separate residual + restriction passes (M4 item 3).
template<class T>
void gmgResidRestrictDevice(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    GmgFab<T>& crhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha
)
{
    for (amrex::MFIter mfi(crhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto cr = crhs.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int ic, int jc, int kc) noexcept
            {
                T acc = 0;
                for (int dk = 0; dk < 2; ++dk)
                {
                    for (int dj = 0; dj < 2; ++dj)
                    {
                        for (int di = 0; di < 2; ++di)
                        {
                            const int i = 2 * ic + di, j = 2 * jc + dj, k = 2 * kc + dk;
                            const T aE = ax(i + 1, j, k);
                            const T aW = lxa(i, j, k);
                            const T aN = ay(i, j + 1, k);
                            const T aS = lya(i, j, k);
                            const T aT = az(i, j, k + 1);
                            const T aB = lza(i, j, k);
                            const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                        + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                        + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                            const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                            acc += b(i, j, k) - (diag * psi(i, j, k) + off);
                        }
                    }
                }
                cr(ic, jc, kc) = static_cast<T>(0.125) * acc;
            }
        );
    }
}

template<class T>
void gmgResidRestrictHost(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    GmgFab<T>& crhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha
)
{
    for (amrex::MFIter mfi(crhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto cr = crhs.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int kc = lo.z; kc <= hi.z; ++kc)
        {
            for (int jc = lo.y; jc <= hi.y; ++jc)
            {
                for (int ic = lo.x; ic <= hi.x; ++ic)
                {
                    T acc = 0;
                    for (int dk = 0; dk < 2; ++dk)
                    {
                        for (int dj = 0; dj < 2; ++dj)
                        {
                            for (int di = 0; di < 2; ++di)
                            {
                                const int i = 2 * ic + di, j = 2 * jc + dj, k = 2 * kc + dk;
                                const T aE = ax(i + 1, j, k);
                                const T aW = lxa(i, j, k);
                                const T aN = ay(i, j + 1, k);
                                const T aS = lya(i, j, k);
                                const T aT = az(i, j, k + 1);
                                const T aB = lza(i, j, k);
                                const T off =
                                    aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                    + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                    + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                                const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                                acc += b(i, j, k) - (diag * psi(i, j, k) + off);
                            }
                        }
                    }
                    cr(ic, jc, kc) = static_cast<T>(0.125) * acc;
                }
            }
        }
    }
}

// One fused Jacobi-Chebyshev degree step: computes r = rhs - A sol on the fly
// (sol's ghosts must be filled) and the polynomial increment
// d = cb * D^{-1} r + (readOld ? ca * d : 0), D = alpha - sum(face coeffs). sol
// is NOT written here (its neighbours are read for r) — the caller adds d to sol
// afterwards, so the whole step is Jacobi-like (race-free) and, being a fixed
// polynomial in the symmetric operator, a symmetric linear smoother (CG-safe).
template<class T>
void gmgChebComputeDDevice(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha,
    GmgFab<T>& d,
    T ca,
    T cb,
    bool readOld
)
{
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto dd = d.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const T aE = ax(i + 1, j, k);
                const T aW = lxa(i, j, k);
                const T aN = ay(i, j + 1, k);
                const T aS = lya(i, j, k);
                const T aT = az(i, j, k + 1);
                const T aB = lza(i, j, k);
                const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                            + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                            + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                const T r = b(i, j, k) - (diag * psi(i, j, k) + off);
                T dval = cb * (r / diag);
                if (readOld)
                {
                    dval += ca * dd(i, j, k);
                }
                dd(i, j, k) = dval;
            }
        );
    }
}

template<class T>
void gmgChebComputeDHost(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha,
    GmgFab<T>& d,
    T ca,
    T cb,
    bool readOld
)
{
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto dd = d.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const T aE = ax(i + 1, j, k);
                    const T aW = lxa(i, j, k);
                    const T aN = ay(i, j + 1, k);
                    const T aS = lya(i, j, k);
                    const T aT = az(i, j, k + 1);
                    const T aB = lza(i, j, k);
                    const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    const T r = b(i, j, k) - (diag * psi(i, j, k) + off);
                    T dval = cb * (r / diag);
                    if (readOld)
                    {
                        dval += ca * dd(i, j, k);
                    }
                    dd(i, j, k) = dval;
                }
            }
        }
    }
}

// out = D^{-1} A v (v's ghosts filled), used by the setup power iteration that
// estimates lambda_max of D^{-1}A per level for the Chebyshev interval.
template<class T>
void gmgDinvApplyDevice(
    const GmgFab<T>& v,
    GmgFab<T>& out,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha
)
{
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = v.const_array(mfi);
        const auto o = out.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const T aE = ax(i + 1, j, k);
                const T aW = lxa(i, j, k);
                const T aN = ay(i, j + 1, k);
                const T aS = lya(i, j, k);
                const T aT = az(i, j, k + 1);
                const T aB = lza(i, j, k);
                const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                            + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                            + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                o(i, j, k) = (diag * psi(i, j, k) + off) / diag;
            }
        );
    }
}

template<class T>
void gmgDinvApplyHost(
    const GmgFab<T>& v,
    GmgFab<T>& out,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha
)
{
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = v.const_array(mfi);
        const auto o = out.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const T aE = ax(i + 1, j, k);
                    const T aW = lxa(i, j, k);
                    const T aN = ay(i, j + 1, k);
                    const T aS = lya(i, j, k);
                    const T aT = az(i, j, k + 1);
                    const T aB = lza(i, j, k);
                    const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    o(i, j, k) = (diag * psi(i, j, k) + off) / diag;
                }
            }
        }
    }
}

// Checkerboard seed (+-1 by cell parity) for the power iteration — close to the
// top eigenvector of the 7-point operator, so few iterations suffice.
template<class T>
void gmgFillCheckerDevice(GmgFab<T>& v)
{
    for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = v.array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { a(i, j, k) = (((i + j + k) & 1) == 0) ? T(1) : T(-1); }
        );
    }
}

template<class T>
void gmgFillCheckerHost(GmgFab<T>& v)
{
    for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = v.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    a(i, j, k) = (((i + j + k) & 1) == 0) ? T(1) : T(-1);
                }
            }
        }
    }
}

// One multigrid level: geometry, rediscretised coefficients and preallocated
// work fields (sol needs 1 ghost for the stencil; rhs is valid-only).
template<class T>
struct GmgLevelT
{
    amrex::Geometry geom;
    std::shared_ptr<GmgFab<T>> alpha, ux, lx, uy, ly, uz, lz;
    std::shared_ptr<GmgFab<T>> sol, rhs;
    std::shared_ptr<GmgFab<T>> chebD; // Chebyshev increment (only when smoother="chebyshev")
    double lambdaMax = 0.0;            // estimate of lambda_max(D^{-1}A) on this level
};

// Native matrix-free geometric-multigrid V-cycle preconditioner on the
// face-coefficient operator: z = M^{-1} r via `n_cycles` V-cycles with
// red-black Gauss-Seidel smoothing (the same smoother family MLMG uses;
// measured much stronger than damped Jacobi here: 9/9 vs 16/16 CG iterations
// at N=32/64 with omega=6/7 Jacobi, 20/22 with omega=2/3), volume-average
// restriction and piecewise-constant prolongation. The V-cycle is symmetric —
// the post-smoother runs the colours in REVERSED order (black-red), making it
// the adjoint of the pre-smoother, and prolongation is the adjoint of
// restriction up to a constant — so it is CG-safe. The whole hierarchy is
// built ONCE at construction — no per-apply allocation; the coefficients are
// copied, so later in-place updates to the caller's fields are seen by the
// outer operator but not by this preconditioner (a slightly stale
// preconditioner only costs iterations).
// Abstract hook exposing a GMG V-cycle as operations on FP64 MultiFabs, so the
// native stationary solver (FaceCoeffSolver solver="gmg") can drive the
// precision-templated GmgPrecondT<T> without knowing T. The whole apply runs on
// AMReX fabs (no Ginkgo vector), converting FP64<->T at the two ends. M3 fuses the
// FP64 residual, its convert-scatter into the (T-typed) L0 rhs and the FP64 norm
// into one kernel (residScatterNorm); vcycleGather runs the V-cycle(s) and adds
// the correction back onto the FP64 x.
class GmgApplyMf
{
public:

    virtual ~GmgApplyMf() = default;

    // Fused r = rhs - A*sol - shift -> (cast to T) L0 rhs; L0 sol := 0; returns the
    // FP64 sum of squares of r (norm authority stays double even for a float L0
    // rhs). `sol`'s ghosts must already be filled by the caller.
    virtual double residScatterNorm(
        const amrex::MultiFab& sol,
        const amrex::MultiFab& rhs,
        const amrex::MultiFab& ux,
        const amrex::MultiFab& lx,
        const amrex::MultiFab& uy,
        const amrex::MultiFab& ly,
        const amrex::MultiFab& uz,
        const amrex::MultiFab& lz,
        const amrex::MultiFab& alpha,
        double shift
    ) const = 0;

    // Run nCycles_ V-cycles on the L0 rhs set by residScatterNorm, then x += the
    // (converted) L0 correction.
    virtual void vcycleGather(amrex::MultiFab& x) const = 0;
};

template<class T>
class GmgPrecondT :
    public gko::EnableLinOp<GmgPrecondT<T>>,
    public gko::EnableCreateMethod<GmgPrecondT<T>>,
    public GmgApplyMf
{
public:

    explicit GmgPrecondT(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<GmgPrecondT<T>>(exec)
    {}

    GmgPrecondT(
        std::shared_ptr<const gko::Executor> exec,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        amrex::Geometry geom,
        gko::size_type n,
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        BcArray bc,
        int n_cycles,
        int pre_sweeps,
        int post_sweeps,
        int coarsest_sweeps,
        int max_levels,
        int min_bottom,
        const std::string& smoother
    )
        : gko::EnableLinOp<GmgPrecondT<T>>(exec, gko::dim<2> {n, n}), bc_(bc),
          hasPhysBc_(std::any_of(bc.begin(), bc.end(), [](int b) { return b != 0; })),
          onDevice_(exec->get_master().get() != exec.get()), nCycles_(n_cycles),
          preSweeps_(pre_sweeps), postSweeps_(post_sweeps), coarsestSweeps_(coarsest_sweeps),
          useCheb_(smoother == "chebyshev")
    {
        if (smoother != "rbgs" && smoother != "chebyshev")
        {
            throw std::runtime_error(
                "GmgPrecond: unknown gmg_smoother '" + smoother + "' (expected 'rbgs' or 'chebyshev')"
            );
        }
        // Finest level: copy the coefficients into this preconditioner's arena
        // (default/device on cuda, pinned on reference — MultiFab::Copy handles
        // the cross-arena transfer, cf. pinnedCopy).
        levels_.push_back(makeLevel(ba, dm, geom));
        copyCoeff(*levels_[0].alpha, *alpha);
        copyCoeff(*levels_[0].ux, *ux);
        copyCoeff(*levels_[0].lx, *lx);
        copyCoeff(*levels_[0].uy, *uy);
        copyCoeff(*levels_[0].ly, *ly);
        copyCoeff(*levels_[0].uz, *uz);
        copyCoeff(*levels_[0].lz, *lz);

        // Coarsen by 2 while every box dimension stays divisible by 2 (with
        // >= 2 cells left) and the coarse domain keeps >= 4 cells per
        // direction. The coarse coefficients are rediscretised from the fine
        // ones: face coeff = mean of the 4 covered fine face coeffs / 4
        // (a ~ -beta/dx^2: beta averaged, dx doubled), alpha (per-volume
        // source) = mean of the 8 fine cell values.
        while (true)
        {
            if (max_levels > 0 && static_cast<int>(levels_.size()) >= max_levels)
            {
                break;
            }
            const GmgLevelT<T>& f = levels_.back();
            const amrex::BoxArray& fba = f.alpha->boxArray();
            if (!fba.coarsenable(2, 2))
            {
                break;
            }
            const amrex::Box cdom = amrex::coarsen(f.geom.Domain(), 2);
            if (cdom.shortside() < min_bottom)
            {
                break;
            }
            amrex::BoxArray cba = fba;
            cba.coarsen(2);
            const amrex::Geometry cgeom(
                cdom,
                f.geom.ProbDomain(),
                f.geom.Coord(),
                {f.geom.isPeriodic(0), f.geom.isPeriodic(1), f.geom.isPeriodic(2)}
            );
            levels_.push_back(makeLevel(cba, dm, cgeom));
            GmgLevelT<T>& c = levels_.back();
            const GmgLevelT<T>& fl = levels_[levels_.size() - 2];
            if (onDevice_)
            {
                gmgRestrictDevice(*fl.alpha, *c.alpha);
                gmgCoarsenFaceDevice(*fl.ux, *c.ux, 0, 4.0);
                gmgCoarsenFaceDevice(*fl.lx, *c.lx, 0, 4.0);
                gmgCoarsenFaceDevice(*fl.uy, *c.uy, 1, 4.0);
                gmgCoarsenFaceDevice(*fl.ly, *c.ly, 1, 4.0);
                gmgCoarsenFaceDevice(*fl.uz, *c.uz, 2, 4.0);
                gmgCoarsenFaceDevice(*fl.lz, *c.lz, 2, 4.0);
            }
            else
            {
                gmgRestrictHost(*fl.alpha, *c.alpha);
                gmgCoarsenFaceHost(*fl.ux, *c.ux, 0, 4.0);
                gmgCoarsenFaceHost(*fl.lx, *c.lx, 0, 4.0);
                gmgCoarsenFaceHost(*fl.uy, *c.uy, 1, 4.0);
                gmgCoarsenFaceHost(*fl.ly, *c.ly, 1, 4.0);
                gmgCoarsenFaceHost(*fl.uz, *c.uz, 2, 4.0);
                gmgCoarsenFaceHost(*fl.lz, *c.lz, 2, 4.0);
            }
        }
        amrex::Gpu::streamSynchronize();

        // Chebyshev setup: per level allocate the polynomial increment field and
        // estimate lambda_max(D^{-1}A) via ~15 power iterations (setup-time cost).
        if (useCheb_)
        {
            for (auto& L : levels_)
            {
                L.chebD = makeMf(L.alpha->boxArray(), L.alpha->DistributionMap(), 0);
            }
            for (std::size_t l = 0; l < levels_.size(); ++l)
            {
                levels_[l].lambdaMax = estimateLambdaMax(l);
            }
            amrex::Gpu::streamSynchronize();
        }
    }

    // Native stationary-solver hooks (M1 + M3). residScatterNorm forms the FP64
    // residual and, in the SAME kernel, casts it into the T-typed L0 rhs and
    // reduces its FP64 norm — no separate FP64 residual MultiFab, norm pass, or
    // convert-scatter. vcycleGather then runs the V-cycle(s) and adds the T-typed
    // correction back onto the FP64 x. Runs entirely on AMReX fabs (no Ginkgo
    // vector); conversions are identities when T==double.
    double residScatterNorm(
        const amrex::MultiFab& sol,
        const amrex::MultiFab& rhs,
        const amrex::MultiFab& ux,
        const amrex::MultiFab& lx,
        const amrex::MultiFab& uy,
        const amrex::MultiFab& ly,
        const amrex::MultiFab& uz,
        const amrex::MultiFab& lz,
        const amrex::MultiFab& alpha,
        double shift
    ) const override
    {
        const GmgLevelT<T>& L0 = levels_.front();
        double sumsq;
        if (onDevice_)
        {
            sumsq = faceCoeffResidScatterNormDevice<T>(
                sol, rhs, ux, lx, uy, ly, uz, lz, alpha, shift, *L0.rhs
            );
            L0.sol->setVal(T(0)); // z0 = 0: apply M^{-1}, not a warm-started solve
        }
        else
        {
            sumsq = faceCoeffResidScatterNormHost<T>(
                sol, rhs, ux, lx, uy, ly, uz, lz, alpha, shift, *L0.rhs
            );
            L0.sol->setVal(T(0));
            amrex::Gpu::streamSynchronize();
        }
        return sumsq;
    }

    void vcycleGather(amrex::MultiFab& x) const override
    {
        const GmgLevelT<T>& L0 = levels_.front();
        if (onDevice_)
        {
            {
                prof::Timer t("gmg.vcycle");
                for (int c = 0; c < nCycles_; ++c)
                {
                    vcycle(0);
                }
            }
            {
                prof::Timer t("gmg.solve.gather");
                gmgConvertAddDevice(x, *L0.sol); // x += (double) L0 correction
                amrex::Gpu::streamSynchronize();
            }
        }
        else
        {
            for (int c = 0; c < nCycles_; ++c)
            {
                vcycle(0);
            }
            gmgConvertAddHost(x, *L0.sol);
            amrex::Gpu::streamSynchronize();
        }
    }

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto exec = this->get_executor();
        const GmgLevelT<T>& L0 = levels_.front();
        if (onDevice_)
        {
            prof::Timer tAll("gmg.apply");
            {
                prof::Timer t("gmg.sync_gko");
                exec->synchronize(); // b written by Ginkgo
            }
            {
                prof::Timer t("gmg.scatter");
                scatter_device(gko::as<Dense>(b)->get_const_values(), *L0.rhs);
                L0.sol->setVal(0.0); // z0 = 0: apply M^{-1}, not a warm-started solve
            }
            {
                prof::Timer t("gmg.vcycle");
                for (int c = 0; c < nCycles_; ++c)
                {
                    vcycle(0);
                }
            }
            {
                prof::Timer t("gmg.gather");
                gather_device(*L0.sol, gko::as<Dense>(x)->get_values(), 1.0);
                amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
            }
        }
        else
        {
            auto host = exec->get_master();
            auto bHost = gko::clone(host, gko::as<Dense>(b));
            scatter(bHost->get_const_values(), *L0.rhs);
            L0.sol->setVal(0.0);
            amrex::Gpu::streamSynchronize(); // setVal may run on the GPU stream
            for (int c = 0; c < nCycles_; ++c)
            {
                vcycle(0);
            }
            auto xHost = Dense::create(host, gko::as<Dense>(x)->get_size());
            gather(*L0.sol, xHost->get_values(), 1.0);
            gko::as<Dense>(x)->copy_from(xHost);
        }
    }

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        auto denseX = gko::as<Dense>(x);
        auto tmp = denseX->clone();
        this->apply_impl(b, tmp.get());
        denseX->scale(beta);
        denseX->add_scaled(alpha, tmp);
    }

private:

    // Chebyshev smooths modes with eigenvalue in [lambdaMax / kChebEigRatio,
    // lambdaMax]; the lower modes are left to the coarse grid. alpha ~= 4-8 is
    // the usual band; 6 minimised the CG count here (degree-2 -> 11 iters at
    // N=32/64 vs rbgs 9, a sweep over {2,3,4,6,8,15,30} at setup).
    static constexpr double kChebEigRatio = 6.0;
    static constexpr double kChebSafety = 1.05; // inflate the lambda_max estimate
    static constexpr int kPowerIters = 15;      // power iterations for lambda_max

    std::shared_ptr<GmgFab<T>> makeMf(
        const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int ng
    ) const
    {
        auto mf = onDevice_
                    ? std::make_shared<GmgFab<T>>(ba, dm, 1, ng)
                    : std::make_shared<GmgFab<T>>(
                          ba, dm, 1, ng, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
                      );
        mf->setVal(T(0));
        return mf;
    }

    GmgLevelT<T> makeLevel(
        const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
        const amrex::Geometry& geom
    ) const
    {
        GmgLevelT<T> L;
        L.geom = geom;
        L.alpha = makeMf(ba, dm, 0);
        const auto fba = [&ba](int d)
        { return amrex::convert(ba, amrex::IntVect::TheDimensionVector(d)); };
        L.ux = makeMf(fba(0), dm, 0);
        L.lx = makeMf(fba(0), dm, 0);
        L.uy = makeMf(fba(1), dm, 0);
        L.ly = makeMf(fba(1), dm, 0);
        L.uz = makeMf(fba(2), dm, 0);
        L.lz = makeMf(fba(2), dm, 0);
        L.sol = makeMf(ba, dm, 1);
        L.rhs = makeMf(ba, dm, 0);
        return L;
    }

    // Copy the caller's FP64 coefficient MultiFab into a level fab, converting
    // to T. On the reference path the source may live in device memory, so it is
    // staged through a pinned FP64 copy before the host conversion loop.
    void copyCoeff(GmgFab<T>& dst, const amrex::MultiFab& src) const
    {
        if (onDevice_)
        {
            gmgConvertCopyDevice(dst, src);
        }
        else
        {
            auto tmp = pinnedCopy(src);
            amrex::Gpu::streamSynchronize();
            gmgConvertCopyHost(dst, *tmp);
        }
    }

    // Fill sol's ghost layer: periodic/internal via FillBoundary, then the
    // homogeneous Dirichlet/Neumann reflection on domain faces (the gap-2 BC
    // fills coarsen cleanly, so the same bc spec applies on every level).
    void fillGhosts(const GmgLevelT<T>& L, int lvl) const
    {
        prof::Timer t("gmg.fill", lvl);
        L.sol->FillBoundary(L.geom.periodicity());
        if (!onDevice_)
        {
            amrex::Gpu::streamSynchronize(); // FillBoundary before host loops
        }
        if (hasPhysBc_)
        {
            if (onDevice_)
            {
                fillDomainBcGhostsDevice(*L.sol, L.geom.Domain(), bc_);
            }
            else
            {
                fillDomainBcGhostsHost(*L.sol, L.geom.Domain(), bc_);
            }
        }
    }

    // Dispatch to the configured smoother. `reversed` is only meaningful for
    // red-black Gauss-Seidel (post-smoother runs the colours in reversed order,
    // the adjoint of the forward sweep); Chebyshev is symmetric by construction
    // so it ignores it. `sweeps` is the RB-GS sweep count / the Chebyshev degree.
    void smooth(std::size_t l, int sweeps, bool reversed) const
    {
        if (useCheb_)
        {
            chebyshevSmooth(l, sweeps);
        }
        else
        {
            rbgsSmooth(l, sweeps, reversed);
        }
    }

    // Red-black Gauss-Seidel sweeps; `reversed` flips the colour order
    // (black-red), which is the adjoint of the forward sweep — used for the
    // post-smoother so the whole V-cycle is symmetric.
    void rbgsSmooth(std::size_t l, int sweeps, bool reversed) const
    {
        const GmgLevelT<T>& L = levels_[l];
        for (int s = 0; s < sweeps; ++s)
        {
            for (int c = 0; c < 2; ++c)
            {
                const int parity = (reversed ? 1 + c : c) & 1;
                fillGhosts(L, static_cast<int>(l)); // the other colour changed — refresh ghosts
                if (onDevice_)
                {
                    prof::Timer t("gmg.gs", static_cast<int>(l));
                    gmgGsColorDevice(
                        *L.sol, *L.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha, parity
                    );
                }
                else
                {
                    gmgGsColorHost(
                        *L.sol, *L.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha, parity
                    );
                }
            }
        }
    }

    // Jacobi-preconditioned Chebyshev smoother of degree `degree`: one full-cell
    // fused residual+increment kernel per degree (plain-stencil bandwidth, no
    // colour split, one ghost fill per degree). A fixed polynomial in the
    // symmetric operator -> symmetric linear smoother, CG-safe by construction.
    void chebyshevSmooth(std::size_t l, int degree) const
    {
        if (degree <= 0)
        {
            return;
        }
        const GmgLevelT<T>& L = levels_[l];
        const double b = L.lambdaMax;
        const double a = b / kChebEigRatio;
        const double theta = 0.5 * (b + a);
        const double delta = 0.5 * (b - a);
        const double sigma = theta / delta;
        double rho = 1.0 / sigma;
        for (int m = 0; m < degree; ++m)
        {
            fillGhosts(L, static_cast<int>(l));
            double ca = 0.0;
            double cb = 0.0;
            bool readOld = false;
            if (m == 0)
            {
                cb = 1.0 / theta; // d = (1/theta) D^{-1} r
            }
            else
            {
                const double rhoNew = 1.0 / (2.0 * sigma - rho);
                ca = rho * rhoNew;         // d = ca * d + cb * D^{-1} r
                cb = 2.0 * rhoNew / delta;
                readOld = true;
                rho = rhoNew;
            }
            if (onDevice_)
            {
                prof::Timer t("gmg.cheb", static_cast<int>(l));
                gmgChebComputeDDevice(
                    *L.sol, *L.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha, *L.chebD,
                    static_cast<T>(ca), static_cast<T>(cb), readOld
                );
            }
            else
            {
                gmgChebComputeDHost(
                    *L.sol, *L.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha, *L.chebD,
                    static_cast<T>(ca), static_cast<T>(cb), readOld
                );
                amrex::Gpu::streamSynchronize();
            }
            GmgFab<T>::Saxpy(*L.sol, T(1), *L.chebD, 0, 0, 1, amrex::IntVect(0)); // sol += d
            if (!onDevice_)
            {
                amrex::Gpu::streamSynchronize();
            }
        }
    }

    // lambda_max(D^{-1}A) on level l via power iteration on a checkerboard seed
    // (near the top eigenvector). Returns the estimate inflated by kChebSafety
    // so the Chebyshev interval upper bound is not undershot.
    double estimateLambdaMax(std::size_t l) const
    {
        const GmgLevelT<T>& L = levels_[l];
        GmgFab<T>& v = *L.sol;    // scratch (1 ghost)
        GmgFab<T>& w = *L.chebD;  // scratch (0 ghost)
        if (onDevice_)
        {
            gmgFillCheckerDevice(v);
        }
        else
        {
            gmgFillCheckerHost(v);
            amrex::Gpu::streamSynchronize();
        }
        double norm = gmgNorm2(v);
        v.mult(static_cast<T>(1.0 / norm), 0, 1, 0);
        double lambda = 0.0;
        for (int it = 0; it < kPowerIters; ++it)
        {
            fillGhosts(L, static_cast<int>(l));
            if (onDevice_)
            {
                gmgDinvApplyDevice(v, w, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha);
            }
            else
            {
                gmgDinvApplyHost(v, w, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha);
                amrex::Gpu::streamSynchronize();
            }
            lambda = gmgNorm2(w); // v is unit-norm -> ||D^{-1}A v|| ~ lambda_max
            if (lambda <= 0.0)
            {
                break;
            }
            if (onDevice_)
            {
                gmgConvertCopyDevice(v, w); // v <- w
            }
            else
            {
                gmgConvertCopyHost(v, w);
                amrex::Gpu::streamSynchronize();
            }
            v.mult(static_cast<T>(1.0 / lambda), 0, 1, 0);
        }
        v.setVal(T(0)); // leave sol clean for the V-cycle
        amrex::Gpu::streamSynchronize();
        return lambda * kChebSafety;
    }

    // One V-cycle correcting levels_[l].sol in place (warm start allowed, so
    // repeated cycles at l = 0 compose correctly).
    void vcycle(std::size_t l) const
    {
        const GmgLevelT<T>& L = levels_[l];
        if (l + 1 == levels_.size())
        {
            // Tiny grid: smoothing is cheap; forward + reversed halves keep
            // the coarsest "solve" self-adjoint (RB-GS; Chebyshev is symmetric
            // regardless, so the two halves just compose into a degree-2*n poly).
            smooth(l, coarsestSweeps_ / 2, false);
            smooth(l, coarsestSweeps_ / 2, true);
            return;
        }
        smooth(l, preSweeps_, false);
        fillGhosts(L, static_cast<int>(l));
        const GmgLevelT<T>& C = levels_[l + 1];
        // Fused residual + restriction: coarse rhs = avg(rhs - A sol) computed on
        // the fly, saving the separate fine-grid residual read+write (M4 item 3).
        {
            prof::Timer t("gmg.residrestrict", static_cast<int>(l));
            if (onDevice_)
            {
                gmgResidRestrictDevice(
                    *L.sol, *L.rhs, *C.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha
                );
            }
            else
            {
                gmgResidRestrictHost(
                    *L.sol, *L.rhs, *C.rhs, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha
                );
            }
            C.sol->setVal(0.0);
        }
        if (!onDevice_)
        {
            amrex::Gpu::streamSynchronize(); // setVal before host loops
        }
        vcycle(l + 1);
        {
            prof::Timer t("gmg.prolong", static_cast<int>(l));
            if (onDevice_)
            {
                gmgProlongAddDevice(*C.sol, *L.sol);
            }
            else
            {
                gmgProlongAddHost(*C.sol, *L.sol);
            }
        }
        smooth(l, postSweeps_, true);
    }

    BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool onDevice_ = false;
    int nCycles_ = 1;
    int preSweeps_ = 2;
    int postSweeps_ = 2;
    int coarsestSweeps_ = 8;
    bool useCheb_ = false;
    std::vector<GmgLevelT<T>> levels_;
};

// One long-lived CudaExecutor per process (see the note in ginkgo_solve): a
// per-call executor re-inits cuBLAS/cuSPARSE and disturbs AMReX's CUDA context
// at teardown. Assumes a single AMReX Initialize/Finalize cycle.
std::shared_ptr<const gko::Executor> makeExecutor(const std::string& executor)
{
    if (executor == "reference")
    {
        return gko::ReferenceExecutor::create();
    }
    if (executor == "cuda")
    {
        static std::shared_ptr<gko::CudaExecutor> cudaExec =
            gko::CudaExecutor::create(0, gko::ReferenceExecutor::create());
        return cudaExec;
    }
    throw std::runtime_error("ginkgo: unknown executor '" + executor + "'");
}

// Per-iteration residual-norm history. Ginkgo's iteration_complete event
// hands (solver, b, x, it, residual, residual_norm, implicit_sq_norm, ...);
// the criteria used here make the solvers pass residual_norm = nullptr, so
// the norm is computed from the residual vector (with the implicit squared
// norm as a last resort). Scalars land on the solve executor, so device
// values are staged through the host master before reading.
class ResidualHistoryLogger : public gko::log::Logger
{
public:

    ResidualHistoryLogger() : gko::log::Logger(gko::log::Logger::iteration_complete_mask) {}

    void clear() { history_.clear(); }

    const std::vector<double>& history() const { return history_; }

protected:

    void on_iteration_complete(
        const gko::LinOp*,
        const gko::LinOp*,
        const gko::LinOp*,
        const gko::size_type&,
        const gko::LinOp* residual,
        const gko::LinOp* residual_norm,
        const gko::LinOp* implicit_sq_norm,
        const gko::array<gko::stopping_status>*,
        bool
    ) const override
    {
        if (auto norm = dynamic_cast<const Dense*>(residual_norm))
        {
            history_.push_back(readScalar(norm));
        }
        else if (auto res = dynamic_cast<const Dense*>(residual))
        {
            auto exec = res->get_executor();
            auto norm2 = Dense::create(exec, gko::dim<2> {1, 1});
            res->compute_norm2(norm2);
            history_.push_back(readScalar(norm2.get()));
        }
        else if (auto sq = dynamic_cast<const Dense*>(implicit_sq_norm))
        {
            history_.push_back(std::sqrt(std::abs(readScalar(sq))));
        }
    }

private:

    static double readScalar(const Dense* d)
    {
        auto exec = d->get_executor();
        if (exec->get_master().get() != exec.get())
        {
            auto host = gko::clone(exec->get_master(), d);
            return host->at(0, 0);
        }
        return d->at(0, 0);
    }

    mutable std::vector<double> history_;
};

// Build a Krylov solver over `op`, stopping on iteration count, the relative
// residual ||r|| <= rtol*||rhs|| (recomputed per solve, so one generate() is
// reused across right-hand sides), or — when atol > 0 — the absolute residual
// ||r|| <= atol. A non-null `precond` (an already-generated LinOp, e.g.
// MlmgPrecond) is attached as the solver's generated preconditioner.
std::shared_ptr<gko::LinOp> buildKrylov(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    int max_iter,
    double rtol,
    double atol,
    std::shared_ptr<const gko::LinOp> precond = nullptr
)
{
    std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> criteria;
    criteria.push_back(
        gko::stop::Iteration::build().with_max_iters(static_cast<gko::size_type>(max_iter)).on(exec)
    );
    criteria.push_back(gko::stop::ResidualNorm<double>::build()
                           .with_baseline(gko::stop::mode::rhs_norm)
                           .with_reduction_factor(rtol)
                           .on(exec));
    if (atol > 0.0)
    {
        criteria.push_back(gko::stop::ResidualNorm<double>::build()
                               .with_baseline(gko::stop::mode::absolute)
                               .with_reduction_factor(atol)
                               .on(exec));
    }
    if (solver == "cg")
    {
        auto params = gko::solver::Cg<double>::build().with_criteria(criteria);
        if (precond)
        {
            params.with_generated_preconditioner(precond);
        }
        return params.on(exec)->generate(op);
    }
    if (solver == "bicgstab")
    {
        auto params = gko::solver::Bicgstab<double>::build().with_criteria(criteria);
        if (precond)
        {
            params.with_generated_preconditioner(precond);
        }
        return params.on(exec)->generate(op);
    }
    if (solver == "gmres")
    {
        auto params = gko::solver::Gmres<double>::build().with_criteria(criteria);
        if (precond)
        {
            params.with_generated_preconditioner(precond);
        }
        return params.on(exec)->generate(op);
    }
    if (solver == "ir")
    {
        // Iterative refinement x <- x + relax * S(b - A x), where S is the
        // already-generated inner solver `precond` (the GMG V-cycle LinOp). With
        // relaxation_factor 1.0 this is plain Richardson driven by the V-cycle,
        // Ginkgo's idiomatic counterpart of the native solver="gmg" loop.
        // default_initial_guess defaults to `provided`, so the incoming x seeds
        // the iteration (the persistent-solver warm-start contract).
        auto params = gko::solver::Ir<double>::build().with_criteria(criteria);
        params.with_relaxation_factor(1.0);
        if (precond)
        {
            params.with_generated_solver(precond);
        }
        return params.on(exec)->generate(op);
    }
    throw std::runtime_error("ginkgo: unknown solver '" + solver + "'");
}

// Assemble the face-coefficient matrix into a CSR on `exec`. SINGLE-BOX
// periodic only (matches the benchmark meshes): neighbour column indices wrap
// around the domain, and the row/column order is the same idx(i,j,k) =
// (k*nj + j)*ni + i used by the gather/scatter pack. This is the assembled
// counterpart of FaceCoeffOp, for measuring the matrix-free advantage.
std::shared_ptr<gko::matrix::Csr<double, int>> assembleFaceCoeffCsr(
    std::shared_ptr<const gko::Executor> exec,
    const amrex::Geometry& geom,
    const amrex::MultiFab& alpha,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz
)
{
    if (alpha.size() != 1)
    {
        throw std::runtime_error("assembleFaceCoeffCsr: single-box meshes only");
    }
    const amrex::Box dom = geom.Domain();
    const int ni = dom.length(0);
    const int nj = dom.length(1);
    const int nk = dom.length(2);
    const long n = static_cast<long>(ni) * nj * nk;

    // Host-accessible copies to read the (device) coefficients.
    auto al = pinnedCopy(alpha);
    auto axu = pinnedCopy(ux);
    auto axl = pinnedCopy(lx);
    auto ayu = pinnedCopy(uy);
    auto ayl = pinnedCopy(ly);
    auto azu = pinnedCopy(uz);
    auto azl = pinnedCopy(lz);
    amrex::Gpu::streamSynchronize();

    amrex::MFIter mfi(*al);
    const auto A = al->const_array(mfi);
    const auto Ux = axu->const_array(mfi);
    const auto Lx = axl->const_array(mfi);
    const auto Uy = ayu->const_array(mfi);
    const auto Ly = ayl->const_array(mfi);
    const auto Uz = azu->const_array(mfi);
    const auto Lz = azl->const_array(mfi);
    const auto lo = amrex::lbound(mfi.validbox());

    std::vector<int> row_ptrs(static_cast<std::size_t>(n) + 1);
    std::vector<int> col_idxs;
    std::vector<double> values;
    col_idxs.reserve(static_cast<std::size_t>(7 * n));
    values.reserve(static_cast<std::size_t>(7 * n));

    auto idx = [=](int i, int j, int k) { return (static_cast<long>(k) * nj + j) * ni + i; };

    row_ptrs[0] = 0;
    for (int k = 0; k < nk; ++k)
    {
        for (int j = 0; j < nj; ++j)
        {
            for (int i = 0; i < ni; ++i)
            {
                const int ia = lo.x + i, ja = lo.y + j, ka = lo.z + k;
                const double aE = Ux(ia + 1, ja, ka);
                const double aW = Lx(ia, ja, ka);
                const double aN = Uy(ia, ja + 1, ka);
                const double aS = Ly(ia, ja, ka);
                const double aT = Uz(ia, ja, ka + 1);
                const double aB = Lz(ia, ja, ka);
                const double diag = A(ia, ja, ka) - (aE + aW + aN + aS + aT + aB);

                // 7 stencil entries (col, val), sorted by column for the row.
                std::array<std::pair<long, double>, 7> e = {
                    {{idx(i, j, (k - 1 + nk) % nk), aB},
                     {idx(i, (j - 1 + nj) % nj, k), aS},
                     {idx((i - 1 + ni) % ni, j, k), aW},
                     {idx(i, j, k), diag},
                     {idx((i + 1) % ni, j, k), aE},
                     {idx(i, (j + 1) % nj, k), aN},
                     {idx(i, j, (k + 1) % nk), aT}}
                };
                std::sort(
                    e.begin(),
                    e.end(),
                    [](const auto& p, const auto& q) { return p.first < q.first; }
                );
                for (const auto& [c, v] : e)
                {
                    col_idxs.push_back(static_cast<int>(c));
                    values.push_back(v);
                }
                row_ptrs[static_cast<std::size_t>(idx(i, j, k)) + 1] =
                    static_cast<int>(col_idxs.size());
            }
        }
    }

    return gko::share(gko::matrix::Csr<double, int>::create(
        exec,
        gko::dim<2> {static_cast<gko::size_type>(n), static_cast<gko::size_type>(n)},
        gko::array<double>(exec, values.begin(), values.end()),
        gko::array<int>(exec, col_idxs.begin(), col_idxs.end()),
        gko::array<int>(exec, row_ptrs.begin(), row_ptrs.end())
    ));
}

// Persistent solver: the operator, the generated Ginkgo solver and the device
// scratch vectors are built ONCE; each solve is just pack rhs -> apply ->
// unpack sol, reusing everything (no per-call operator/solver rebuild). The
// concrete operator is supplied by a subclass.
class PersistentSolver
{
public:

    virtual ~PersistentSolver() = default;

    virtual nb::dict solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
    {
        resLogger_->clear(); // per-call history
        {
            prof::Timer t("solve.pack");
            if (onDevice_)
            {
                gather_device(rhs, b_->get_values(), 1.0);
                gather_device(sol, x_->get_values(), 1.0);
                amrex::Gpu::streamSynchronize();
            }
            else
            {
                gather(rhs, b_->get_values(), 1.0);
                gather(sol, x_->get_values(), 1.0);
            }
        }

        if (projectNullspace_)
        {
            // Singular system with the constant nullspace (e.g. fully-periodic
            // pure Poisson): make the rhs consistent by removing its mean, and
            // keep the initial guess in the mean-zero subspace so CG stays there.
            subtractMean(b_.get());
            subtractMean(x_.get());
        }

        {
            prof::Timer t("solve.krylov");
            solver_->apply(b_, x_);
        }

        if (projectNullspace_)
        {
            // Pin the arbitrary constant: return the mean-zero representative
            // (also removes any roundoff drift out of the subspace).
            subtractMean(x_.get());
        }

        {
            prof::Timer t("solve.unpack");
            if (onDevice_)
            {
                exec_->synchronize();
                scatter_device(x_->get_const_values(), sol);
                amrex::Gpu::streamSynchronize();
            }
            else
            {
                scatter(x_->get_const_values(), sol);
            }
        }

        // Final 2-norm residual ||b - A x|| for reporting.
        prof::Timer tRep("solve.report");
        auto res = b_->clone();
        auto one = gko::initialize<Dense>({1.0}, exec_);
        auto negOne = gko::initialize<Dense>({-1.0}, exec_);
        op_->apply(negOne, x_, one, res);
        auto norm = Dense::create(exec_, gko::dim<2> {1, 1});
        res->compute_norm2(norm);
        auto normHost = gko::clone(exec_->get_master(), norm);

        nb::dict d;
        d["num_iters"] = static_cast<std::int64_t>(logger_->get_num_iterations());
        d["res_norm"] = normHost->at(0, 0);
        d["converged"] = logger_->has_converged();
        nb::list hist;
        for (double v : resLogger_->history())
        {
            hist.append(v);
        }
        d["res_history"] = hist;
        return d;
    }

protected:

    // allocDense=false skips the n-sized Ginkgo work vectors b_/x_ — the native
    // stationary solver (solver="gmg") drives the V-cycle on MultiFabs and never
    // touches them (a real memory saving at large N: 2 * n doubles).
    PersistentSolver(std::shared_ptr<const gko::Executor> exec, gko::size_type n, bool allocDense = true)
        : exec_(std::move(exec)), onDevice_(exec_->get_master().get() != exec_.get()), n_(n)
    {
        if (allocDense)
        {
            b_ = Dense::create(exec_, gko::dim<2> {n_, 1});
            x_ = Dense::create(exec_, gko::dim<2> {n_, 1});
        }
    }

    // Subclass calls this once its operator is built.
    void build(
        std::shared_ptr<gko::LinOp> op,
        const std::string& solver,
        int max_iter,
        double rtol,
        double atol,
        bool project_nullspace,
        std::shared_ptr<const gko::LinOp> precond = nullptr
    )
    {
        op_ = std::move(op);
        solver_ = buildKrylov(solver, exec_, op_, max_iter, rtol, atol, std::move(precond));
        logger_ = gko::share(gko::log::Convergence<double>::create());
        solver_->add_logger(logger_);
        resLogger_ = std::make_shared<ResidualHistoryLogger>();
        solver_->add_logger(resLogger_);
        projectNullspace_ = project_nullspace;
        if (projectNullspace_)
        {
            ones_ = Dense::create(exec_, gko::dim<2> {n_, 1});
            ones_->fill(1.0);
        }
    }

    // v -= mean(v), computed on the executor (dot with ones); only the scalar
    // mean crosses to the host. Uniform cells, so volume mean == arithmetic mean.
    void subtractMean(Dense* v)
    {
        auto sum = Dense::create(exec_, gko::dim<2> {1, 1});
        v->compute_dot(ones_, sum);
        auto sumHost = gko::clone(exec_->get_master(), sum);
        auto negMean =
            gko::initialize<Dense>({-sumHost->at(0, 0) / static_cast<double>(n_)}, exec_);
        v->add_scaled(negMean, ones_);
    }

    std::shared_ptr<const gko::Executor> exec_;
    bool onDevice_;
    gko::size_type n_;
    std::shared_ptr<gko::LinOp> op_;
    std::unique_ptr<Dense> b_;
    std::unique_ptr<Dense> x_;
    std::shared_ptr<gko::LinOp> solver_;
    std::shared_ptr<gko::log::Convergence<double>> logger_;
    std::shared_ptr<ResidualHistoryLogger> resLogger_;
    bool projectNullspace_ = false;
    std::unique_ptr<Dense> ones_;
};

// Matrix-free persistent solver: the operator reads the caller's coefficient
// fields on the fly, so an external in-place update to them changes the matrix
// with no reassembly.
class FaceCoeffSolver : public PersistentSolver
{
public:

    FaceCoeffSolver(
        const std::string& executor,
        amrex::Geometry geom,
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        const std::string& solver,
        int max_iter,
        double rtol,
        double atol,
        bool project_nullspace,
        MLMG* precond_mlmg,
        int precond_cycles,
        const std::vector<std::string>& bc,
        const std::string& precond,
        int gmg_pre_sweeps,
        int gmg_post_sweeps,
        int gmg_coarsest_sweeps,
        int gmg_max_levels,
        int gmg_min_bottom,
        const std::string& gmg_smoother,
        const std::string& gmg_precision
    )
        : PersistentSolver(
              makeExecutor(executor), static_cast<gko::size_type>(alpha->boxArray().numPts()),
              solver != "gmg"
          )
    {
        // CG-safety: the V-cycle is a symmetric (SPD) preconditioner only when
        // the post-smoother is the adjoint of the pre-smoother, which requires
        // equal pre/post counts. With asymmetric counts CG's assumption breaks;
        // warn but allow (usable as a stationary/flexible-CG smoother). The native
        // stationary solver (solver="gmg") is NOT CG, so asymmetric sweeps there
        // are legitimate and never warn (this guard requires solver=="cg").
        if (precond == "gmg" && solver == "cg" && gmg_pre_sweeps != gmg_post_sweeps)
        {
            std::cerr << "FaceCoeffSolver: warning — gmg_pre_sweeps ("
                      << gmg_pre_sweeps << ") != gmg_post_sweeps (" << gmg_post_sweeps
                      << ") makes the V-cycle non-symmetric; CG may stall or diverge. "
                         "Use equal counts for a CG-safe preconditioner.\n";
        }
        const BcArray bcArr = parseBc(bc, geom, "FaceCoeffSolver");

        // solver="gmg": native stationary geometric-multigrid solver
        // (x <- x + V(b - A x) until tolerance). The GMG V-cycle IS the solver,
        // so `precond` is ignored; the hierarchy is built directly and the whole
        // iteration runs on AMReX fabs (see gmgSolve). No Ginkgo Krylov object.
        if (solver == "gmg")
        {
            if (precond_mlmg != nullptr)
            {
                throw std::runtime_error(
                    "FaceCoeffSolver: solver='gmg' cannot be combined with precond_mlmg"
                );
            }
            gmgStationary_ = true;
            if (onDevice_)
            {
                // Device residual kernel reads the caller's device coefficients
                // directly (in-place updates are seen, like FaceCoeffOp).
                alpha_ = alpha;
                ux_ = ux;
                lx_ = lx;
                uy_ = uy;
                ly_ = ly;
                uz_ = uz;
                lz_ = lz;
            }
            else
            {
                // Host residual loops can't read device memory: stage the
                // coefficients to pinned once (solve-constant, cf. FaceCoeffOp).
                ownedCoeff_ = {
                    pinnedCopy(*alpha),
                    pinnedCopy(*ux),
                    pinnedCopy(*lx),
                    pinnedCopy(*uy),
                    pinnedCopy(*ly),
                    pinnedCopy(*uz),
                    pinnedCopy(*lz)
                };
                alpha_ = ownedCoeff_[0].get();
                ux_ = ownedCoeff_[1].get();
                lx_ = ownedCoeff_[2].get();
                uy_ = ownedCoeff_[3].get();
                ly_ = ownedCoeff_[4].get();
                uz_ = ownedCoeff_[5].get();
                lz_ = ownedCoeff_[6].get();
            }
            geom_ = geom;
            bcArr_ = bcArr;
            hasPhysBc_ = std::any_of(bcArr.begin(), bcArr.end(), [](int b) { return b != 0; });
            maxIter_ = max_iter;
            rtol_ = rtol;
            atol_ = atol;
            projectNull_ = project_nullspace;
            gmgOwner_ = buildGmgHierarchy(
                alpha, ux, lx, uy, ly, uz, lz, geom, bcArr, precond_cycles, gmg_pre_sweeps,
                gmg_post_sweeps, gmg_coarsest_sweeps, gmg_max_levels, gmg_min_bottom, gmg_smoother,
                gmg_precision
            );
            const amrex::BoxArray& ba = alpha->boxArray();
            const amrex::DistributionMapping& dm = alpha->DistributionMap();
            if (onDevice_)
            {
                xWork_ = std::make_shared<amrex::MultiFab>(ba, dm, 1, 1);
            }
            else
            {
                xWork_ = std::make_shared<amrex::MultiFab>(
                    ba, dm, 1, 1, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
                );
                rhsPinned_ = std::make_shared<amrex::MultiFab>(
                    ba, dm, 1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
                );
            }
            return;
        }

        auto op = gko::share(FaceCoeffOp::create(
            exec_,
            alpha->boxArray(),
            alpha->DistributionMap(),
            geom,
            n_,
            alpha,
            ux,
            lx,
            uy,
            ly,
            uz,
            lz,
            bcArr
        ));

        // solver="ir": Ginkgo iterative refinement (gko::solver::Ir<double>) whose
        // system matrix is the FaceCoeffOp above and whose inner solver is the
        // generated GMG V-cycle LinOp (with_generated_solver, relaxation 1.0). Like
        // solver="gmg" it implies the GMG hierarchy and ignores `precond`; unlike it
        // the loop runs through Ginkgo (Dense pack/unpack + Convergence logger kept),
        // so the measured overhead across the LinOp boundaries vs the native gmg loop
        // is part of the deliverable — this variant does NOT fuse across it.
        if (solver == "ir")
        {
            if (precond_mlmg != nullptr)
            {
                throw std::runtime_error(
                    "FaceCoeffSolver: solver='ir' cannot be combined with precond_mlmg"
                );
            }
            auto inner = buildGmgHierarchy(
                alpha, ux, lx, uy, ly, uz, lz, geom, bcArr, precond_cycles, gmg_pre_sweeps,
                gmg_post_sweeps, gmg_coarsest_sweeps, gmg_max_levels, gmg_min_bottom, gmg_smoother,
                gmg_precision
            );
            build(op, solver, max_iter, rtol, atol, project_nullspace, std::move(inner));
            return;
        }

        std::shared_ptr<const gko::LinOp> pc;
        if (precond == "gmg")
        {
            if (precond_mlmg != nullptr)
            {
                throw std::runtime_error(
                    "FaceCoeffSolver: precond='gmg' cannot be combined with precond_mlmg"
                );
            }
            pc = buildGmgHierarchy(
                alpha, ux, lx, uy, ly, uz, lz, geom, bcArr, precond_cycles, gmg_pre_sweeps,
                gmg_post_sweeps, gmg_coarsest_sweeps, gmg_max_levels, gmg_min_bottom, gmg_smoother,
                gmg_precision
            );
        }
        else if (precond == "mlmg" || precond == "none")
        {
            // precond_mlmg alone implies "mlmg" (pre-existing behaviour).
            if (precond == "mlmg" && precond_mlmg == nullptr)
            {
                throw std::runtime_error("FaceCoeffSolver: precond='mlmg' requires precond_mlmg");
            }
            if (precond_mlmg != nullptr)
            {
                pc = gko::share(MlmgPrecond::create(
                    exec_,
                    precond_mlmg,
                    alpha->boxArray(),
                    alpha->DistributionMap(),
                    n_,
                    precond_cycles
                ));
            }
        }
        else
        {
            throw std::runtime_error(
                "FaceCoeffSolver: unknown precond '" + precond
                + "' (expected 'none', 'mlmg' or 'gmg')"
            );
        }
        build(op, solver, max_iter, rtol, atol, project_nullspace, std::move(pc));
    }

    // Native stationary GMG solver (solver="gmg") drives the V-cycle on MultiFabs;
    // every other solver keeps the base Krylov path. Dispatch here so the binding
    // (which calls S::solve on the concrete type) picks the right loop.
    nb::dict solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) override
    {
        if (gmgStationary_)
        {
            return gmgSolve(rhs, sol);
        }
        return PersistentSolver::solve(rhs, sol);
    }

private:

    // Build the precision-templated V-cycle hierarchy (fp64 default — byte-for-
    // byte the historical behaviour; fp32 halves the bandwidth-bound V-cycle
    // bytes, outer residual stays fp64). Also records the GmgApplyMf* so the
    // stationary solver can drive the V-cycle on fabs without knowing the type.
    std::shared_ptr<const gko::LinOp> buildGmgHierarchy(
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        const amrex::Geometry& geom,
        const BcArray& bcArr,
        int precond_cycles,
        int gmg_pre_sweeps,
        int gmg_post_sweeps,
        int gmg_coarsest_sweeps,
        int gmg_max_levels,
        int gmg_min_bottom,
        const std::string& gmg_smoother,
        const std::string& gmg_precision
    )
    {
        if (gmg_precision != "fp64" && gmg_precision != "fp32")
        {
            throw std::runtime_error(
                "FaceCoeffSolver: unknown gmg_precision '" + gmg_precision
                + "' (expected 'fp64' or 'fp32')"
            );
        }
        auto makeGmg = [&](auto tag) -> std::shared_ptr<const gko::LinOp>
        {
            using T = decltype(tag);
            auto p = GmgPrecondT<T>::create(
                exec_, alpha->boxArray(), alpha->DistributionMap(), geom, n_, alpha, ux, lx, uy, ly,
                uz, lz, bcArr, precond_cycles, gmg_pre_sweeps, gmg_post_sweeps, gmg_coarsest_sweeps,
                gmg_max_levels, gmg_min_bottom, gmg_smoother
            );
            gmgMf_ = p.get(); // GmgPrecondT<T>* -> const GmgApplyMf* (kept alive by the return)
            return gko::share(std::move(p));
        };
        return (gmg_precision == "fp32") ? makeGmg(float {}) : makeGmg(double {});
    }

    // Fill xWork_'s ghost layer for the FP64 residual: periodic/internal via
    // FillBoundary, then homogeneous domain BCs via ghost reflection — the same
    // fill FaceCoeffOp does, so the residual uses the identical operator A.
    void fillGmgGhosts(amrex::MultiFab& mf) const
    {
        mf.FillBoundary(geom_.periodicity());
        if (!onDevice_)
        {
            amrex::Gpu::streamSynchronize();
        }
        if (hasPhysBc_)
        {
            if (onDevice_)
            {
                fillDomainBcGhostsDevice(mf, geom_.Domain(), bcArr_);
            }
            else
            {
                fillDomainBcGhostsHost(mf, geom_.Domain(), bcArr_);
            }
        }
    }

    // mf -= mean(mf) over the valid region (constant-nullspace projection for
    // singular systems; uniform cells so the volume mean is the arithmetic mean).
    void subtractMeanMf(amrex::MultiFab& mf) const
    {
        const double mean = mf.sum(0) / static_cast<double>(n_);
        mf.plus(-mean, 0, 1);
    }

    // Native stationary V-cycle solve: x <- x + V(b - A x), warm-started from the
    // incoming sol, until ||r|| <= max(rtol*||b||, atol) or max_iter cycles. Runs
    // entirely on AMReX fabs — no Ginkgo Krylov object, no per-iteration
    // flat-vector pack/unpack, no per-iteration Ginkgo<->AMReX crossings.
    nb::dict gmgSolve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
    {
        // Warm start: x0 = incoming sol (do NOT zero — persistent-solver contract).
        amrex::MultiFab::Copy(*xWork_, sol, 0, 0, 1, 0);

        // Host residual loops can't read the device rhs: stage it to pinned once
        // per solve (it is constant across the cycle loop). Device path reads rhs
        // directly.
        const amrex::MultiFab* rhsUse = &rhs;
        if (!onDevice_)
        {
            amrex::MultiFab::Copy(*rhsPinned_, rhs, 0, 0, 1, 0);
            amrex::Gpu::streamSynchronize();
            rhsUse = rhsPinned_.get();
        }

        const double bNorm = rhs.norm2(0);
        const double stopTol = std::max(rtol_ * bNorm, atol_);
        const double rhsMean = projectNull_ ? rhs.sum(0) / static_cast<double>(n_) : 0.0;
        if (projectNull_)
        {
            subtractMeanMf(*xWork_);
        }

        std::vector<double> history;
        // M3: one fused kernel forms the FP64 residual r = rhs - A x - rhsMean,
        // casts it into the (fp32/fp64) L0 rhs, and reduces ||r|| in double — no
        // separate FP64 residual MultiFab, norm pass, or convert-scatter. The
        // nullspace shift (rhsMean) folds into the same kernel, so the projected
        // path takes the fused route too (it only adds subtractMeanMf on x).
        auto computeResid = [&]() -> double
        {
            prof::Timer t("gmg.solve.resid");
            fillGmgGhosts(*xWork_);
            const double sumsq = gmgMf_->residScatterNorm(
                *xWork_, *rhsUse, *ux_, *lx_, *uy_, *ly_, *uz_, *lz_, *alpha_, rhsMean
            );
            const double rn = std::sqrt(sumsq);
            history.push_back(rn);
            return rn;
        };

        double rnorm = computeResid();
        bool converged = rnorm <= stopTol;
        int cycles = 0;
        while (!converged && cycles < maxIter_)
        {
            {
                prof::Timer t("gmg.solve.vcycle");
                gmgMf_->vcycleGather(*xWork_); // x += V(r); the residual is already in L0 rhs
            }
            if (projectNull_)
            {
                subtractMeanMf(*xWork_);
            }
            ++cycles;
            rnorm = computeResid();
            converged = rnorm <= stopTol;
        }

        amrex::MultiFab::Copy(sol, *xWork_, 0, 0, 1, 0);

        nb::dict d;
        d["num_iters"] = static_cast<std::int64_t>(cycles);
        d["res_norm"] = rnorm;
        d["converged"] = converged;
        nb::list hist;
        for (double v : history)
        {
            hist.append(v);
        }
        d["res_history"] = hist;
        return d;
    }

    // Native stationary GMG solver state (only populated when solver="gmg").
    bool gmgStationary_ = false;
    const amrex::MultiFab* alpha_ = nullptr;
    const amrex::MultiFab* ux_ = nullptr;
    const amrex::MultiFab* lx_ = nullptr;
    const amrex::MultiFab* uy_ = nullptr;
    const amrex::MultiFab* ly_ = nullptr;
    const amrex::MultiFab* uz_ = nullptr;
    const amrex::MultiFab* lz_ = nullptr;
    amrex::Geometry geom_ {};
    BcArray bcArr_ {};
    bool hasPhysBc_ = false;
    int maxIter_ = 0;
    double rtol_ = 0.0;
    double atol_ = 0.0;
    bool projectNull_ = false;
    std::shared_ptr<const gko::LinOp> gmgOwner_; // keeps the V-cycle hierarchy alive
    const GmgApplyMf* gmgMf_ = nullptr;          // typed V-cycle hook into gmgOwner_
    std::shared_ptr<amrex::MultiFab> xWork_;     // FP64 iterate (1 ghost)
    std::shared_ptr<amrex::MultiFab> rhsPinned_; // pinned rhs stage (reference path)
    std::vector<std::shared_ptr<amrex::MultiFab>> ownedCoeff_; // pinned coeffs (reference path)
};

// Assembled-CSR persistent solver: same matrix, stored explicitly. Its per-
// iteration SpMV streams the matrix from memory, versus FaceCoeffSolver which
// recomputes entries from the face coefficients — the matrix-free comparison.
class FaceCoeffCsrSolver : public PersistentSolver
{
public:

    FaceCoeffCsrSolver(
        const std::string& executor,
        amrex::Geometry geom,
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        const std::string& solver,
        int max_iter,
        double rtol,
        double atol,
        bool project_nullspace,
        MLMG* precond_mlmg,
        int precond_cycles,
        const std::vector<std::string>& bc,
        const std::string& precond,
        int /*gmg_pre_sweeps*/,
        int /*gmg_post_sweeps*/,
        int /*gmg_coarsest_sweeps*/,
        int /*gmg_max_levels*/,
        int /*gmg_min_bottom*/,
        const std::string& /*gmg_smoother*/,
        const std::string& /*gmg_precision*/
    )
        : PersistentSolver(
              makeExecutor(executor), static_cast<gko::size_type>(alpha->boxArray().numPts())
          )
    {
        // The CSR assembly wraps neighbour indices around the domain
        // (periodic-only); parseBc also rejects a non-periodic geometry.
        const BcArray bcArr = parseBc(bc, geom, "FaceCoeffCsrSolver");
        if (std::any_of(bcArr.begin(), bcArr.end(), [](int b) { return b != 0; }))
        {
            throw std::runtime_error(
                "FaceCoeffCsrSolver: periodic boundaries only — use FaceCoeffSolver "
                "for dirichlet/neumann bc"
            );
        }
        if (precond == "gmg")
        {
            throw std::runtime_error(
                "FaceCoeffCsrSolver: precond='gmg' is matrix-free only — use FaceCoeffSolver"
            );
        }
        if (precond != "none" && precond != "mlmg")
        {
            throw std::runtime_error(
                "FaceCoeffCsrSolver: unknown precond '" + precond
                + "' (expected 'none' or 'mlmg')"
            );
        }
        if (precond == "mlmg" && precond_mlmg == nullptr)
        {
            throw std::runtime_error("FaceCoeffCsrSolver: precond='mlmg' requires precond_mlmg");
        }
        auto op = assembleFaceCoeffCsr(exec_, geom, *alpha, *ux, *lx, *uy, *ly, *uz, *lz);
        std::shared_ptr<const gko::LinOp> pc;
        if (precond_mlmg != nullptr)
        {
            pc = gko::share(MlmgPrecond::create(
                exec_, precond_mlmg, alpha->boxArray(), alpha->DistributionMap(), n_, precond_cycles
            ));
        }
        build(op, solver, max_iter, rtol, atol, project_nullspace, std::move(pc));
    }
};

// Bind a persistent solver class S (constructor: coefficients + geom + config;
// method: solve(rhs, sol)). keep_alive ties the coefficient fields to the
// solver, since the matrix-free operator references them on the device.
template<class S>
void bindPersistent(nb::module_& m, const char* name)
{
    nb::class_<S>(m, name)
        .def(
            "__init__",
            [](S* self,
               amrex::MultiFab& alpha,
               amrex::MultiFab& ux,
               amrex::MultiFab& lx,
               amrex::MultiFab& uy,
               amrex::MultiFab& ly,
               amrex::MultiFab& uz,
               amrex::MultiFab& lz,
               const amrex::Geometry& geom,
               const std::string& executor,
               const std::string& solver,
               int max_iter,
               double rtol,
               double atol,
               bool project_nullspace,
               MLMG* precond_mlmg,
               int precond_cycles,
               const std::vector<std::string>& bc,
               const std::string& precond,
               int gmg_pre_sweeps,
               int gmg_post_sweeps,
               int gmg_coarsest_sweeps,
               int gmg_max_levels,
               int gmg_min_bottom,
               const std::string& gmg_smoother,
               const std::string& gmg_precision)
            {
                new (self) S(
                    executor, geom, &alpha, &ux, &lx, &uy, &ly, &uz, &lz, solver, max_iter, rtol,
                    atol, project_nullspace, precond_mlmg, precond_cycles, bc, precond,
                    gmg_pre_sweeps, gmg_post_sweeps, gmg_coarsest_sweeps, gmg_max_levels,
                    gmg_min_bottom, gmg_smoother, gmg_precision
                );
            },
            nb::arg("alpha"),
            nb::arg("ux"),
            nb::arg("lx"),
            nb::arg("uy"),
            nb::arg("ly"),
            nb::arg("uz"),
            nb::arg("lz"),
            nb::arg("geom"),
            nb::arg("executor") = "cuda",
            // Krylov solvers "cg" | "bicgstab" | "gmres", OR "gmg" (matrix-free
            // solver only): the NATIVE stationary geometric-multigrid solver
            // x <- x + V(b - A x) run to tolerance (Richardson iteration, like
            // MLMG) — no Ginkgo Krylov object, the whole loop on AMReX fabs.
            // solver="gmg" builds the V-cycle hierarchy directly and IGNORES the
            // `precond` argument (the V-cycle IS the solver). A standalone
            // V-cycle needs the coarsest grid solved accurately, so raise
            // gmg_coarsest_sweeps (~100 for rbgs, ~160 for chebyshev) — the
            // CG-tuned default of 8 gives a weak, slowly-converging iteration.
            // solver="ir" is the Ginkgo-idiomatic twin of "gmg": a
            // gko::solver::Ir<double> (iterative refinement, relaxation 1.0) whose
            // system matrix is the matrix-free FaceCoeffOp and whose inner solver is
            // the generated GMG V-cycle LinOp. Same GMG semantics (builds the
            // hierarchy, ignores `precond`, needs the accurate coarsest solve) but
            // driven through Ginkgo's Dense pack/unpack + Convergence logger.
            nb::arg("solver") = "bicgstab",
            nb::arg("max_iter") = 1000,
            nb::arg("rtol") = 1e-10,
            nb::arg("atol") = 0.0,
            nb::arg("project_nullspace") = false,
            nb::arg("precond_mlmg").none() = nb::none(),
            nb::arg("precond_cycles") = 1,
            // Domain BCs, order (xlo, xhi, ylo, yhi, zlo, zhi); each entry is
            // "periodic", "dirichlet" (homogeneous, u=0 on the face) or
            // "neumann" (homogeneous, du/dn=0). Must match the geometry's
            // periodicity per direction. Matrix-free solver only.
            nb::arg("bc") = std::vector<std::string>(6, "periodic"),
            // Preconditioner selector: "none" (default; precond_mlmg alone
            // implies "mlmg"), "mlmg" (requires precond_mlmg) or "gmg" (native
            // matrix-free geometric multigrid on the face coefficients —
            // matrix-free solver only, no MLMG involved).
            nb::arg("precond") = "none",
            // Native-GMG (precond="gmg") V-cycle knobs. Defaults reproduce the
            // previous fixed behaviour. gmg_pre_sweeps/gmg_post_sweeps: RB-GS
            // sweep count / Chebyshev degree per pre-/post-smooth (keep them
            // equal for a CG-safe symmetric V-cycle). gmg_coarsest_sweeps:
            // smoothing on the bottom level. gmg_max_levels: 0 = auto/unlimited
            // coarsening; else cap the hierarchy depth. gmg_min_bottom: stop
            // coarsening before the domain shortside drops below this.
            // gmg_smoother: "rbgs" (red-black Gauss-Seidel) or "chebyshev"
            // (Jacobi-preconditioned polynomial, plain-stencil bandwidth).
            nb::arg("gmg_pre_sweeps") = 2,
            nb::arg("gmg_post_sweeps") = 2,
            nb::arg("gmg_coarsest_sweeps") = 8,
            nb::arg("gmg_max_levels") = 0,
            nb::arg("gmg_min_bottom") = 4,
            nb::arg("gmg_smoother") = "rbgs",
            // Native-GMG hierarchy precision: "fp64" (default; byte-for-byte the
            // previous behaviour) or "fp32" — the whole V-cycle (level
            // coefficients, work fields, smoother, restriction/prolongation,
            // ghost fills) runs in single precision while the outer CG/operator
            // stays double, halving the bandwidth-bound V-cycle traffic.
            // Matrix-free solver only.
            nb::arg("gmg_precision") = "fp64",
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>(),
            nb::keep_alive<1, 4>(),
            nb::keep_alive<1, 5>(),
            nb::keep_alive<1, 6>(),
            nb::keep_alive<1, 7>(),
            nb::keep_alive<1, 8>(),
            // The preconditioner MLMG (arg 16; self=1, args from 2) must
            // outlive the solver — MlmgPrecond holds a raw pointer to it.
            // keep_alive is a no-op when the arg is None.
            nb::keep_alive<1, 16>()
        )
        .def(
            "solve",
            [](S& self, amrex::MultiFab& rhs, amrex::MultiFab& sol)
            { return self.solve(rhs, sol); },
            nb::arg("rhs"),
            nb::arg("sol"),
            "Solve A sol = rhs, reusing the prebuilt operator and solver. sol's\n"
            "incoming values seed the initial guess; the matrix is defined by the\n"
            "coefficient fields handed to the constructor (and, for the matrix-free\n"
            "solver, re-read each call so in-place updates take effect). With\n"
            "project_nullspace=True (constructor kwarg, for singular systems with\n"
            "the constant nullspace, e.g. fully-periodic pure Poisson) the rhs and\n"
            "initial guess are projected mean-zero before the Krylov solve and the\n"
            "returned solution is the mean-zero representative. With precond_mlmg\n"
            "(constructor kwarg: an MLMG built on an equivalent operator) each\n"
            "Krylov iteration is preconditioned by precond_cycles multigrid\n"
            "V-cycles, keeping the iteration count ~flat in N. precond='gmg'\n"
            "(constructor kwarg, matrix-free solver only) instead uses the\n"
            "native geometric-multigrid V-cycle built directly on the face\n"
            "coefficients (no MLMG anywhere). bc (constructor\n"
            "kwarg, matrix-free solver only): 6 entries (xlo, xhi, ylo, yhi,\n"
            "zlo, zhi) of 'periodic' | 'dirichlet' | 'neumann' — homogeneous\n"
            "domain BCs folded in via ghost reflection; must match the\n"
            "geometry's periodicity per direction. Returns a\n"
            "dict with num_iters, res_norm, converged and res_history (per-\n"
            "iteration residual norms of this call)."
        );
}

} // namespace

void registerGinkgoSolve(nb::module_& m)
{
    using namespace amrex;

    using MLLinOp = MLLinOpT<MultiFab>;

    m.def(
        "ginkgo_solve",
        [](MLLinOp& lp,
           MultiFab& sol,
           const MultiFab& rhs,
           int max_iter,
           double rtol,
           double atol,
           double sign,
           const std::string& executor)
        {
            MLMG mlmg(lp);

            // "reference" keeps the Krylov vector ops on the CPU; "cuda" runs
            // them on the GPU (device 0) with a ReferenceExecutor as host
            // master. The mat-vec (MLMG::apply) is on the GPU either way.
            std::shared_ptr<const gko::Executor> exec;
            if (executor == "reference")
            {
                exec = gko::ReferenceExecutor::create();
            }
            else if (executor == "cuda")
            {
                // One long-lived CudaExecutor for the whole process: creating
                // and destroying it per call re-inits cuBLAS/cuSPARSE each time
                // and, on teardown, disturbs the CUDA primary context AMReX
                // still needs at finalize (CUDA error 709). Kept alive here so
                // finalize sees a live context; assumes a single AMReX
                // Initialize/Finalize cycle (true for the tests and benchmark).
                static std::shared_ptr<gko::CudaExecutor> cudaExec =
                    gko::CudaExecutor::create(0, gko::ReferenceExecutor::create());
                exec = cudaExec;
            }
            else
            {
                throw std::runtime_error("ginkgo_solve: unknown executor '" + executor + "'");
            }
            const BoxArray& ba = sol.boxArray();
            const DistributionMapping& dm = sol.DistributionMap();
            const auto n = static_cast<gko::size_type>(ba.numPts());

            // Op construction runs one apply to record c0 = L_inhom(0).
            auto op = gko::share(AmrexOp::create(exec, &mlmg, ba, dm, n, sign));

            // r0 = rhs - L_inhom(x0), x0 = incoming sol. MLMG::apply needs a
            // ghost cell on the input (and overwrites it), so copy sol's valid
            // region into a zero-initialized scratch rather than passing sol.
            MultiFab scratch(ba, dm, 1, 1, MFInfo().SetArena(The_Pinned_Arena()));
            scratch.setVal(0.0);
            MultiFab::Copy(scratch, sol, 0, 0, 1, 0);
            MultiFab r0(ba, dm, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));
            mlmg.apply({&r0}, {&scratch});
            // Xpay: dst = src + a*dst, i.e. r0 = rhs - L_inhom(x0).
            MultiFab::Xpay(r0, -1.0, rhs, 0, 0, 1, 0);

            // b = sign*r0, matching the sign inside AmrexOp; the correction
            // delta starts at zero.
            // gather writes host-side; build b on the executor's host master,
            // then move it to the (possibly device) solver executor.
            auto bHost = Dense::create(exec->get_master(), gko::dim<2> {n, 1});
            gather(r0, bHost->get_values(), sign);
            auto b = gko::clone(exec, bHost);
            auto x = Dense::create(exec, gko::dim<2> {n, 1});
            x->fill(0.0);

            // Stop on ||r_k|| <= rtol * ||rhs|| of the ORIGINAL system (an
            // absolute criterion here): the correction system's own rhs is
            // sign*r0, and relative to that a warm start (tiny r0) would grind
            // to reduce an already-converged residual by another factor rtol.
            // The correction residual equals the original-system residual, so
            // atol > 0 adds the plain absolute stop ||r_k|| <= atol.
            const double rhsNorm = rhs.norm2(0);
            const double stopTol = (rhsNorm > 0.0) ? rtol * rhsNorm : rtol;
            std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> criteria;
            criteria.push_back(gko::stop::Iteration::build()
                                   .with_max_iters(static_cast<gko::size_type>(max_iter))
                                   .on(exec));
            criteria.push_back(gko::stop::ResidualNorm<double>::build()
                                   .with_baseline(gko::stop::mode::absolute)
                                   .with_reduction_factor(stopTol)
                                   .on(exec));
            if (atol > 0.0)
            {
                criteria.push_back(gko::stop::ResidualNorm<double>::build()
                                       .with_baseline(gko::stop::mode::absolute)
                                       .with_reduction_factor(atol)
                                       .on(exec));
            }
            auto logger = gko::share(gko::log::Convergence<double>::create());
            auto resLogger = std::make_shared<ResidualHistoryLogger>();
            auto solver =
                gko::solver::Cg<double>::build().with_criteria(criteria).on(exec)->generate(op);
            solver->add_logger(logger);
            solver->add_logger(resLogger);
            solver->apply(b, x);

            // sol = x0 + delta.
            MultiFab delta(ba, dm, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));
            auto xHost = gko::clone(exec->get_master(), x);
            scatter(xHost->get_const_values(), delta);
            MultiFab::Add(sol, delta, 0, 0, 1, 0);

            // Explicit final residual ||b - A_home delta||_2 for reporting.
            auto res = b->clone();
            auto one = gko::initialize<Dense>({1.0}, exec);
            auto negOne = gko::initialize<Dense>({-1.0}, exec);
            op->apply(negOne, x, one, res);
            auto norm = Dense::create(exec, gko::dim<2> {1, 1});
            res->compute_norm2(norm);
            auto normHost = gko::clone(exec->get_master(), norm);

            nb::dict result;
            result["num_iters"] = static_cast<std::int64_t>(logger->get_num_iterations());
            result["res_norm"] = normHost->at(0, 0);
            result["converged"] = logger->has_converged();
            nb::list hist;
            for (double v : resLogger->history())
            {
                hist.append(v);
            }
            result["res_history"] = hist;
            return result;
        },
        nb::arg("lp"),
        nb::arg("sol"),
        nb::arg("rhs"),
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        nb::arg("atol") = 0.0,
        nb::arg("sign") = -1.0,
        nb::arg("executor") = "reference",
        "Matrix-free Ginkgo CG solve of the MLLinOp system L(sol) = rhs.\n\n"
        "sol's incoming values are the initial guess, and boundary data set\n"
        "via set_level_bc is honored (residual-correction solve). `sign` must\n"
        "make sign*L SPD: -1.0 (default) for MLPoisson (L = +laplacian,\n"
        "negative-definite); +1.0 for MLABecLaplacian (alpha*a*phi -\n"
        "beta*div(b grad phi), positive-definite). CG stops when\n"
        "||r_k|| <= rtol*||rhs|| (or ||r_k|| <= atol when atol > 0), so a warm\n"
        "start converges immediately.\n"
        "`executor` is 'reference' (CPU, default) or 'cuda' (GPU device 0). On\n"
        "'cuda' the entire solve runs on the device: the Krylov vector ops, the\n"
        "MLMG::apply mat-vec, and the vector<->MultiFab pack/unpack kernels all\n"
        "stay on the GPU, with no per-iteration host transfer. Returns a dict\n"
        "with num_iters, res_norm (2-norm of the homogeneous-system residual),\n"
        "converged and res_history (per-iteration residual norms)."
    );

    m.def(
        "ginkgo_solve_composite",
        [](MLLinOp& lp,
           nb::list sol_py,
           nb::list rhs_py,
           int max_iter,
           double rtol,
           double atol,
           double sign,
           const std::string& executor,
           const std::string& solver)
        {
            const int nlevs = lp.NAMRLevels();
            if (static_cast<int>(nb::len(sol_py)) != nlevs
                || static_cast<int>(nb::len(rhs_py)) != nlevs)
            {
                throw std::runtime_error(
                    "ginkgo_solve_composite: sol and rhs need one MultiFab per AMR level ("
                    + std::to_string(nlevs) + ")"
                );
            }
            Vector<MultiFab*> sol(nlevs);
            Vector<MultiFab const*> rhs(nlevs);
            for (int lev = 0; lev < nlevs; ++lev)
            {
                sol[lev] = &nb::cast<MultiFab&>(sol_py[static_cast<std::size_t>(lev)]);
                rhs[lev] = &nb::cast<MultiFab const&>(rhs_py[static_cast<std::size_t>(lev)]);
            }

            MLMG mlmg(lp);

            auto exec = makeExecutor(executor);

            std::vector<BoxArray> bas;
            std::vector<DistributionMapping> dms;
            std::vector<long> off;
            long ntot = 0;
            for (int lev = 0; lev < nlevs; ++lev)
            {
                bas.push_back(sol[lev]->boxArray());
                dms.push_back(sol[lev]->DistributionMap());
                off.push_back(ntot);
                ntot += bas.back().numPts();
            }
            const auto n = static_cast<gko::size_type>(ntot);

            // Op construction runs one apply to record c0 = L_inhom(0).
            auto op = gko::share(CompositeAmrexOp::create(exec, &mlmg, bas, dms, n, sign));

            // Refinement ratio between AMR levels lev and lev+1, from the
            // level domains (MLLinOp::AMRRefRatio is protected here).
            auto refRatio = [&lp](int lev)
            {
                const Box& cd = lp.Geom(lev).Domain();
                const Box& fd = lp.Geom(lev + 1).Domain();
                return IntVect(
                    fd.length(0) / cd.length(0),
                    fd.length(1) / cd.length(1),
                    fd.length(2) / cd.length(2)
                );
            };

            // Consistent rhs: coarse cells covered by a finer level are slaved
            // (their operator columns are zero — see CompositeAmrexOp), so
            // their rhs entries must be the average_down of the fine rhs for
            // the system to be solvable. Pinned copies; caller's rhs untouched.
            Vector<MultiFab> rhsC(nlevs);
            for (int lev = 0; lev < nlevs; ++lev)
            {
                rhsC[lev].define(
                    bas[static_cast<std::size_t>(lev)],
                    dms[static_cast<std::size_t>(lev)],
                    1,
                    0,
                    MFInfo().SetArena(The_Pinned_Arena())
                );
                MultiFab::Copy(rhsC[lev], *rhs[lev], 0, 0, 1, 0);
            }
            for (int lev = nlevs - 2; lev >= 0; --lev)
            {
                average_down(rhsC[lev + 1], rhsC[lev], 0, 1, refRatio(lev));
            }

            // r0 = rhs - L_inhom(x0), x0 = incoming sol (per level). MLMG::apply
            // needs a ghost cell on the input (and overwrites it), so copy sol's
            // valid region into zero-initialized scratch rather than passing sol.
            Vector<MultiFab> scratch(nlevs), r0(nlevs);
            Vector<MultiFab*> scratchP(nlevs), r0P(nlevs);
            for (int lev = 0; lev < nlevs; ++lev)
            {
                scratch[lev].define(
                    bas[static_cast<std::size_t>(lev)],
                    dms[static_cast<std::size_t>(lev)],
                    1,
                    1,
                    MFInfo().SetArena(The_Pinned_Arena())
                );
                scratch[lev].setVal(0.0);
                MultiFab::Copy(scratch[lev], *sol[lev], 0, 0, 1, 0);
                r0[lev].define(
                    bas[static_cast<std::size_t>(lev)],
                    dms[static_cast<std::size_t>(lev)],
                    1,
                    0,
                    MFInfo().SetArena(The_Pinned_Arena())
                );
                scratchP[lev] = &scratch[lev];
                r0P[lev] = &r0[lev];
            }
            mlmg.apply(r0P, scratchP);
            for (int lev = 0; lev < nlevs; ++lev)
            {
                // Xpay: dst = src + a*dst, i.e. r0 = rhsC - L_inhom(x0).
                MultiFab::Xpay(r0[lev], -1.0, rhsC[lev], 0, 0, 1, 0);
            }

            // b = sign*r0 packed level-by-level; the correction delta starts
            // at zero.
            auto bHost = Dense::create(exec->get_master(), gko::dim<2> {n, 1});
            for (int lev = 0; lev < nlevs; ++lev)
            {
                gather(r0[lev], bHost->get_values() + off[static_cast<std::size_t>(lev)], sign);
            }
            auto b = gko::clone(exec, bHost);
            auto x = Dense::create(exec, gko::dim<2> {n, 1});
            x->fill(0.0);

            // Stop on the composite ||rhs|| of the ORIGINAL system, as an
            // absolute criterion (see ginkgo_solve for the warm-start rationale).
            double rhsNorm2 = 0.0;
            for (int lev = 0; lev < nlevs; ++lev)
            {
                const double nl = rhsC[lev].norm2(0);
                rhsNorm2 += nl * nl;
            }
            const double rhsNorm = std::sqrt(rhsNorm2);
            const double stopTol = (rhsNorm > 0.0) ? rtol * rhsNorm : rtol;
            std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> criteria;
            criteria.push_back(gko::stop::Iteration::build()
                                   .with_max_iters(static_cast<gko::size_type>(max_iter))
                                   .on(exec));
            criteria.push_back(gko::stop::ResidualNorm<double>::build()
                                   .with_baseline(gko::stop::mode::absolute)
                                   .with_reduction_factor(stopTol)
                                   .on(exec));
            if (atol > 0.0)
            {
                criteria.push_back(gko::stop::ResidualNorm<double>::build()
                                       .with_baseline(gko::stop::mode::absolute)
                                       .with_reduction_factor(atol)
                                       .on(exec));
            }
            auto logger = gko::share(gko::log::Convergence<double>::create());
            auto resLogger = std::make_shared<ResidualHistoryLogger>();
            std::shared_ptr<gko::LinOp> gsolver;
            if (solver == "cg")
            {
                gsolver =
                    gko::solver::Cg<double>::build().with_criteria(criteria).on(exec)->generate(op);
            }
            else if (solver == "bicgstab")
            {
                gsolver = gko::solver::Bicgstab<double>::build()
                              .with_criteria(criteria)
                              .on(exec)
                              ->generate(op);
            }
            else if (solver == "gmres")
            {
                gsolver =
                    gko::solver::Gmres<double>::build().with_criteria(criteria).on(exec)->generate(
                        op
                    );
            }
            else
            {
                throw std::runtime_error("ginkgo_solve_composite: unknown solver '" + solver + "'");
            }
            gsolver->add_logger(logger);
            gsolver->add_logger(resLogger);
            gsolver->apply(b, x);

            // sol = x0 + delta per level, then enforce the covered-cell
            // convention: coarse covered cells = average_down of the fine
            // solution (matching MLMG::solve — the covered entries of x are
            // Krylov by-products, not DOFs).
            auto xHost = gko::clone(exec->get_master(), x);
            for (int lev = 0; lev < nlevs; ++lev)
            {
                MultiFab delta(
                    bas[static_cast<std::size_t>(lev)],
                    dms[static_cast<std::size_t>(lev)],
                    1,
                    0,
                    MFInfo().SetArena(The_Pinned_Arena())
                );
                scatter(xHost->get_const_values() + off[static_cast<std::size_t>(lev)], delta);
                MultiFab::Add(*sol[lev], delta, 0, 0, 1, 0);
            }
            for (int lev = nlevs - 2; lev >= 0; --lev)
            {
                average_down(*sol[lev + 1], *sol[lev], 0, 1, refRatio(lev));
            }
            amrex::Gpu::streamSynchronize();

            // Explicit final residual ||b - A_home delta||_2 for reporting.
            auto res = b->clone();
            auto one = gko::initialize<Dense>({1.0}, exec);
            auto negOne = gko::initialize<Dense>({-1.0}, exec);
            op->apply(negOne, x, one, res);
            auto norm = Dense::create(exec, gko::dim<2> {1, 1});
            res->compute_norm2(norm);
            auto normHost = gko::clone(exec->get_master(), norm);

            nb::dict result;
            result["num_iters"] = static_cast<std::int64_t>(logger->get_num_iterations());
            result["res_norm"] = normHost->at(0, 0);
            result["converged"] = logger->has_converged();
            nb::list hist;
            for (double v : resLogger->history())
            {
                hist.append(v);
            }
            result["res_history"] = hist;
            return result;
        },
        nb::arg("lp"),
        nb::arg("sol"),
        nb::arg("rhs"),
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        nb::arg("atol") = 0.0,
        nb::arg("sign") = -1.0,
        nb::arg("executor") = "reference",
        nb::arg("solver") = "bicgstab",
        "Matrix-free Ginkgo solve of the multi-level COMPOSITE MLLinOp system\n"
        "L(sol) = rhs on a 2+ level AMR hierarchy (one sol/rhs MultiFab per\n"
        "level, coarsest first). The mat-vec is the multi-level MLMG::apply:\n"
        "coarse/fine interface interpolation, reflux and covered-cell\n"
        "average_down are all handled by AMReX, so the solved system is\n"
        "identical to MLMG's own composite solve. Covered coarse cells are\n"
        "slaved, not DOFs: their rhs entries are replaced internally by the\n"
        "average_down of the fine rhs, and on return they hold the\n"
        "average_down of the fine solution. sol's incoming values are the\n"
        "initial guess (residual-correction form, set_level_bc honored).\n"
        "`sign` as in ginkgo_solve: -1.0 for MLPoisson, +1.0 for\n"
        "MLABecLaplacian. The composite operator is not exactly symmetric\n"
        "(c/f interpolation vs reflux), so solver='bicgstab' (default) or\n"
        "'gmres' are safe; 'cg' may work in practice. Stops when\n"
        "||r_k|| <= rtol*||rhs|| (composite norm; or ||r_k|| <= atol when\n"
        "atol > 0). executor='reference'|'cuda'. Returns a dict with\n"
        "num_iters, res_norm, converged and res_history."
    );

    m.def(
        "ginkgo_solve_face_coeffs",
        [](MultiFab& alpha,
           MultiFab& ux,
           MultiFab& lx,
           MultiFab& uy,
           MultiFab& ly,
           MultiFab& uz,
           MultiFab& lz,
           MultiFab& sol,
           const MultiFab& rhs,
           const Geometry& geom,
           const std::string& solver,
           int max_iter,
           double rtol)
        {
            auto exec = gko::ReferenceExecutor::create();
            const BoxArray& ba = sol.boxArray();
            const DistributionMapping& dm = sol.DistributionMap();
            const auto n = static_cast<gko::size_type>(ba.numPts());

            auto op = gko::share(
                FaceCoeffOp::create(exec, ba, dm, geom, n, &alpha, &ux, &lx, &uy, &ly, &uz, &lz)
            );

            // Plain linear solve A x = b: the face coefficients are the full
            // (BC-folded) matrix, so no affine offset. Incoming sol seeds the
            // initial guess (Ginkgo uses x's initial values), rhs is b.
            auto b = Dense::create(exec, gko::dim<2> {n, 1});
            gather(rhs, b->get_values(), 1.0);
            auto x = Dense::create(exec, gko::dim<2> {n, 1});
            gather(sol, x->get_values(), 1.0);

            const double rhsNorm = rhs.norm2(0);
            const double stopTol = (rhsNorm > 0.0) ? rtol * rhsNorm : rtol;

            std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> criteria;
            criteria.push_back(gko::stop::Iteration::build()
                                   .with_max_iters(static_cast<gko::size_type>(max_iter))
                                   .on(exec));
            criteria.push_back(gko::stop::ResidualNorm<double>::build()
                                   .with_baseline(gko::stop::mode::absolute)
                                   .with_reduction_factor(stopTol)
                                   .on(exec));

            auto logger = gko::share(gko::log::Convergence<double>::create());
            std::shared_ptr<gko::LinOp> gsolver;
            if (solver == "cg")
            {
                gsolver =
                    gko::solver::Cg<double>::build().with_criteria(criteria).on(exec)->generate(op);
            }
            else if (solver == "bicgstab")
            {
                gsolver = gko::solver::Bicgstab<double>::build()
                              .with_criteria(criteria)
                              .on(exec)
                              ->generate(op);
            }
            else if (solver == "gmres")
            {
                gsolver =
                    gko::solver::Gmres<double>::build().with_criteria(criteria).on(exec)->generate(
                        op
                    );
            }
            else
            {
                throw std::runtime_error(
                    "ginkgo_solve_face_coeffs: unknown solver '" + solver + "'"
                );
            }
            gsolver->add_logger(logger);
            gsolver->apply(b, x);

            scatter(x->get_const_values(), sol);

            // Explicit final residual ||b - A x||_2 for reporting.
            auto res = b->clone();
            auto one = gko::initialize<Dense>({1.0}, exec);
            auto negOne = gko::initialize<Dense>({-1.0}, exec);
            op->apply(negOne, x, one, res);
            auto norm = Dense::create(exec, gko::dim<2> {1, 1});
            res->compute_norm2(norm);

            nb::dict result;
            result["num_iters"] = static_cast<std::int64_t>(logger->get_num_iterations());
            result["res_norm"] = norm->at(0, 0);
            return result;
        },
        nb::arg("alpha"),
        nb::arg("ux"),
        nb::arg("lx"),
        nb::arg("uy"),
        nb::arg("ly"),
        nb::arg("uz"),
        nb::arg("lz"),
        nb::arg("sol"),
        nb::arg("rhs"),
        nb::arg("geom"),
        nb::arg("solver") = "bicgstab",
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        "Matrix-free Ginkgo solve of a general structured face-coefficient system A(sol) = rhs.\n\n"
        "The matrix is carried as OpenFOAM-style AMReX fields: alpha is the\n"
        "cell-centred diagonal SOURCE (ddt/Sp/reaction), and u{x,y,z}/l{x,y,z}\n"
        "are the face-centred upper/lower off-diagonal coefficients (pass the\n"
        "same field for u* and l* for a symmetric matrix). The full diagonal is\n"
        "assembled on the fly as alpha - negSumDiag(faces). `solver` is one of\n"
        "'cg' (SPD only), 'bicgstab' (default), or 'gmres'. sol's incoming\n"
        "values seed the initial guess. CG/BiCGStab/GMRES stop when\n"
        "||r_k|| <= rtol*||rhs||. Returns a dict with num_iters and res_norm."
    );

    // Persistent solvers: build the operator + Ginkgo solver once, solve many
    // times. FaceCoeffSolver is matrix-free (recomputes the mat-vec from the
    // face coefficients each apply); FaceCoeffCsrSolver assembles the same
    // matrix into a CSR (single-box periodic) so the benefit of matrix-free
    // over an explicit sparse matrix can be measured.
    bindPersistent<FaceCoeffSolver>(m, "FaceCoeffSolver");
    bindPersistent<FaceCoeffCsrSolver>(m, "FaceCoeffCsrSolver");

    // M0 profiling accessors (see namespace prof). Empty unless the process
    // runs with BLOCKAMR_PROFILE=1.
    m.def(
        "profile_report",
        []()
        {
            nb::dict d;
            for (const auto& [key, acc] : prof::table())
            {
                d[key.c_str()] = nb::make_tuple(acc.sec, acc.count);
            }
            return d;
        },
        "Accumulated {phase: (seconds, count)} timers (BLOCKAMR_PROFILE=1)."
    );
    m.def(
        "profile_reset",
        []() { prof::table().clear(); },
        "Clear the BLOCKAMR_PROFILE=1 phase-timer accumulators."
    );
}
