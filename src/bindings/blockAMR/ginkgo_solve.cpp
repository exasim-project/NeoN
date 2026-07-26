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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

// Flat-vector <-> MultiFab transfer (component 0, valid cells only).
// gather and scatter MUST traverse cells in the identical order: MFIter
// without tiling, then k,j,i over the valid box. MultiFabs live in device
// memory by default in GPU builds, so access is staged through explicit
// host copies unless the arena is host-accessible. `scale` lets gather
// apply the SPD sign flip (-L) in the same pass.
void gather(const amrex::MultiFab& mf, double* buf, double scale)
{
    const bool hostOk = mf.arena()->isHostAccessible();
    amrex::Gpu::streamSynchronize();
    std::size_t idx = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto& fab = mf[mfi];
        const amrex::Box& fbx = fab.box();
        std::vector<double> stage;
        auto arr = fab.const_array();
        if (!hostOk)
        {
            // Component 0 occupies the first numPts() elements of the fab.
            stage.resize(static_cast<std::size_t>(fbx.numPts()));
            amrex::Gpu::dtoh_memcpy(stage.data(), fab.dataPtr(), stage.size() * sizeof(double));
            arr = amrex::makeArray4<const double>(stage.data(), fbx, 1);
        }
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    buf[idx++] = scale * arr(i, j, k);
                }
            }
        }
    }
}

void scatter(const double* buf, amrex::MultiFab& mf)
{
    const bool hostOk = mf.arena()->isHostAccessible();
    amrex::Gpu::streamSynchronize();
    std::size_t idx = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        auto& fab = mf[mfi];
        const amrex::Box& fbx = fab.box();
        std::vector<double> stage;
        auto arr = fab.array();
        if (!hostOk)
        {
            // Round-trip the full fab so ghost values survive the update.
            stage.resize(static_cast<std::size_t>(fbx.numPts()));
            amrex::Gpu::dtoh_memcpy(stage.data(), fab.dataPtr(), stage.size() * sizeof(double));
            arr = amrex::makeArray4<double>(stage.data(), fbx, 1);
        }
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    arr(i, j, k) = buf[idx++];
                }
            }
        }
        if (!hostOk)
        {
            amrex::Gpu::htod_memcpy(fab.dataPtr(), stage.data(), stage.size() * sizeof(double));
        }
    }
}

// Device pack/unpack between a contiguous Ginkgo vector (device memory) and a
// device-resident MultiFab, via amrex::ParallelFor so the whole mat-vec runs
// on the GPU with NO host round-trip per Krylov iteration. The flat index MUST
// match the host gather/scatter above (MFIter order; within a valid box the
// index runs fastest in i, then j, then k), because the one-time RHS pack and
// solution unpack in the solve still use the host path.
void scatter_device(const double* vec, amrex::MultiFab& mf)
{
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
                a(i, j, k) = vec[idx];
            }
        );
        off += vbx.numPts();
    }
}

void gather_device(const amrex::MultiFab& mf, double* vec, double scale)
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
                vec[idx] = scale * a(i, j, k);
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
void fillDomainBcGhostsDevice(amrex::MultiFab& mf, const amrex::Box& domain, const BcArray& bc)
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
            const double sign = f.sign;
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
void fillDomainBcGhostsHost(amrex::MultiFab& mf, const amrex::Box& domain, const BcArray& bc)
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

// Device face-coefficient stencil (OpenFOAM Amul, pull form) as a free function
// so the extended __device__ lambda has a namespace-scope enclosing function
// (nvcc forbids it inside a protected/private member). out = A * in with the
// diagonal formed on the fly as alpha - negSumDiag(faces); in's ghosts must
// already be filled.
void faceCoeffStencilDevice(
    const amrex::MultiFab& in,
    amrex::MultiFab& out,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alpha
)
{
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = in.const_array(mfi);
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
                const double off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                 + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                 + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                const double diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                o(i, j, k) = diag * psi(i, j, k) + off;
            }
        );
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
            this->get_executor()->synchronize(); // b written by Ginkgo
            scatter_device(gko::as<Dense>(b)->get_const_values(), *in_);
            in_->FillBoundary(geom_.periodicity());
            if (hasPhysBc_)
            {
                // Domain-boundary ghosts: reflect-odd/even folds the
                // homogeneous Dirichlet/Neumann BCs into the stencil.
                fillDomainBcGhostsDevice(*in_, geom_.Domain(), bc_);
            }
            amrex::Gpu::streamSynchronize();
            // Stencil is a free function: nvcc forbids an extended __device__
            // lambda inside a protected/private member.
            faceCoeffStencilDevice(*in_, *out_, *ux_, *lx_, *uy_, *ly_, *uz_, *lz_, *alpha_);
            gather_device(*out_, gko::as<Dense>(x)->get_values(), 1.0);
            amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
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
// ---------------------------------------------------------------------------

// resid = rhs - A(sol): the negSumDiag face-coefficient stencil in residual
// form. sol's ghosts (periodic + domain BC) must already be filled.
void gmgResidualDevice(
    const amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    amrex::MultiFab& resid,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alpha
)
{
    for (amrex::MFIter mfi(resid); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto r = resid.array(mfi);
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
                const double off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                 + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                 + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                const double diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                r(i, j, k) = b(i, j, k) - (diag * psi(i, j, k) + off);
            }
        );
    }
}

void gmgResidualHost(
    const amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    amrex::MultiFab& resid,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alpha
)
{
    for (amrex::MFIter mfi(resid); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto r = resid.array(mfi);
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
                    const double off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                     + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                     + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const double diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    r(i, j, k) = b(i, j, k) - (diag * psi(i, j, k) + off);
                }
            }
        }
    }
}

// One red-black Gauss-Seidel colour pass: cells with (i+j+k) parity `parity`
// are solved exactly in place, sol = (rhs - off) / D with D = alpha -
// sum(face coeffs) recomputed on the fly (tiny |D| guarded to no update). The
// 7-point stencil only couples opposite colours, so the in-place update is
// race-free. sol's ghosts must be refreshed before EACH colour pass.
void gmgGsColorDevice(
    amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alpha,
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
                if (amrex::Math::abs(diag) > 1e-300)
                {
                    psi(i, j, k) = (b(i, j, k) - off) / diag;
                }
            }
        );
    }
}

void gmgGsColorHost(
    amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alpha,
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
                    if (std::abs(diag) > 1e-300)
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
void gmgRestrictDevice(const amrex::MultiFab& fine, amrex::MultiFab& crse)
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
                c(i, j, k) = 0.125
                           * (f(i2, j2, k2) + f(i2 + 1, j2, k2) + f(i2, j2 + 1, k2)
                              + f(i2 + 1, j2 + 1, k2) + f(i2, j2, k2 + 1) + f(i2 + 1, j2, k2 + 1)
                              + f(i2, j2 + 1, k2 + 1) + f(i2 + 1, j2 + 1, k2 + 1));
            }
        );
    }
}

void gmgRestrictHost(const amrex::MultiFab& fine, amrex::MultiFab& crse)
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
                    c(i, j, k) = 0.125
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
void gmgCoarsenFaceDevice(
    const amrex::MultiFab& fine, amrex::MultiFab& crse, int dir, double scale
)
{
    int u[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    // The two transverse (cell) directions of face-normal `dir`.
    if (dir == 0) { u[1] = 1; v[2] = 1; }
    else if (dir == 1) { u[0] = 1; v[2] = 1; }
    else { u[0] = 1; v[1] = 1; }
    const int u0 = u[0], u1 = u[1], u2 = u[2];
    const int v0 = v[0], v1 = v[1], v2 = v[2];
    const double w = 0.25 / scale;
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

void gmgCoarsenFaceHost(const amrex::MultiFab& fine, amrex::MultiFab& crse, int dir, double scale)
{
    int u[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    if (dir == 0) { u[1] = 1; v[2] = 1; }
    else if (dir == 1) { u[0] = 1; v[2] = 1; }
    else { u[0] = 1; v[1] = 1; }
    const double w = 0.25 / scale;
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
void gmgProlongAddDevice(const amrex::MultiFab& crse, amrex::MultiFab& fine)
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

void gmgProlongAddHost(const amrex::MultiFab& crse, amrex::MultiFab& fine)
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

// One multigrid level: geometry, rediscretised coefficients and preallocated
// work fields (sol needs 1 ghost for the stencil; rhs/resid are valid-only).
struct GmgLevel
{
    amrex::Geometry geom;
    std::shared_ptr<amrex::MultiFab> alpha, ux, lx, uy, ly, uz, lz;
    std::shared_ptr<amrex::MultiFab> sol, rhs, resid;
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
class GmgPrecond : public gko::EnableLinOp<GmgPrecond>, public gko::EnableCreateMethod<GmgPrecond>
{
public:

    explicit GmgPrecond(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<GmgPrecond>(exec)
    {}

    GmgPrecond(
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
        int n_cycles
    )
        : gko::EnableLinOp<GmgPrecond>(exec, gko::dim<2> {n, n}), bc_(bc),
          hasPhysBc_(std::any_of(bc.begin(), bc.end(), [](int b) { return b != 0; })),
          onDevice_(exec->get_master().get() != exec.get()), nCycles_(n_cycles)
    {
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
            const GmgLevel& f = levels_.back();
            const amrex::BoxArray& fba = f.alpha->boxArray();
            if (!fba.coarsenable(2, 2))
            {
                break;
            }
            const amrex::Box cdom = amrex::coarsen(f.geom.Domain(), 2);
            if (cdom.shortside() < 4)
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
            GmgLevel& c = levels_.back();
            const GmgLevel& fl = levels_[levels_.size() - 2];
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
    }

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto exec = this->get_executor();
        const GmgLevel& L0 = levels_.front();
        if (onDevice_)
        {
            exec->synchronize(); // b written by Ginkgo
            scatter_device(gko::as<Dense>(b)->get_const_values(), *L0.rhs);
            L0.sol->setVal(0.0); // z0 = 0: apply M^{-1}, not a warm-started solve
            for (int c = 0; c < nCycles_; ++c)
            {
                vcycle(0);
            }
            gather_device(*L0.sol, gko::as<Dense>(x)->get_values(), 1.0);
            amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
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

    static constexpr int kPreSweeps = 2;  // == kPostSweeps (adjoint order) keeps
    static constexpr int kPostSweeps = 2; // the cycle symmetric for CG
    static constexpr int kCoarsestSweeps = 8; // 4 forward + 4 reversed (self-adjoint)

    std::shared_ptr<amrex::MultiFab> makeMf(
        const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int ng
    ) const
    {
        auto mf = onDevice_
                    ? std::make_shared<amrex::MultiFab>(ba, dm, 1, ng)
                    : std::make_shared<amrex::MultiFab>(
                          ba, dm, 1, ng, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
                      );
        mf->setVal(0.0);
        return mf;
    }

    GmgLevel makeLevel(
        const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
        const amrex::Geometry& geom
    ) const
    {
        GmgLevel L;
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
        L.resid = makeMf(ba, dm, 0);
        return L;
    }

    static void copyCoeff(amrex::MultiFab& dst, const amrex::MultiFab& src)
    {
        amrex::MultiFab::Copy(dst, src, 0, 0, 1, 0);
    }

    // Fill sol's ghost layer: periodic/internal via FillBoundary, then the
    // homogeneous Dirichlet/Neumann reflection on domain faces (the gap-2 BC
    // fills coarsen cleanly, so the same bc spec applies on every level).
    void fillGhosts(const GmgLevel& L) const
    {
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

    void residual(const GmgLevel& L) const // resid = rhs - A sol (ghosts filled)
    {
        if (onDevice_)
        {
            gmgResidualDevice(
                *L.sol, *L.rhs, *L.resid, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha
            );
        }
        else
        {
            gmgResidualHost(
                *L.sol, *L.rhs, *L.resid, *L.ux, *L.lx, *L.uy, *L.ly, *L.uz, *L.lz, *L.alpha
            );
        }
    }

    // Red-black Gauss-Seidel sweeps; `reversed` flips the colour order
    // (black-red), which is the adjoint of the forward sweep — used for the
    // post-smoother so the whole V-cycle is symmetric.
    void smooth(std::size_t l, int sweeps, bool reversed) const
    {
        const GmgLevel& L = levels_[l];
        for (int s = 0; s < sweeps; ++s)
        {
            for (int c = 0; c < 2; ++c)
            {
                const int parity = (reversed ? 1 + c : c) & 1;
                fillGhosts(L); // the other colour changed — refresh ghosts
                if (onDevice_)
                {
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

    // One V-cycle correcting levels_[l].sol in place (warm start allowed, so
    // repeated cycles at l = 0 compose correctly).
    void vcycle(std::size_t l) const
    {
        const GmgLevel& L = levels_[l];
        if (l + 1 == levels_.size())
        {
            // Tiny grid: smoothing is cheap; forward + reversed halves keep
            // the coarsest "solve" self-adjoint.
            smooth(l, kCoarsestSweeps / 2, false);
            smooth(l, kCoarsestSweeps / 2, true);
            return;
        }
        smooth(l, kPreSweeps, false);
        fillGhosts(L);
        residual(L);
        const GmgLevel& C = levels_[l + 1];
        if (onDevice_)
        {
            gmgRestrictDevice(*L.resid, *C.rhs);
        }
        else
        {
            gmgRestrictHost(*L.resid, *C.rhs);
        }
        C.sol->setVal(0.0);
        if (!onDevice_)
        {
            amrex::Gpu::streamSynchronize(); // setVal before host loops
        }
        vcycle(l + 1);
        if (onDevice_)
        {
            gmgProlongAddDevice(*C.sol, *L.sol);
        }
        else
        {
            gmgProlongAddHost(*C.sol, *L.sol);
        }
        smooth(l, kPostSweeps, true);
    }

    BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool onDevice_ = false;
    int nCycles_ = 1;
    std::vector<GmgLevel> levels_;
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

    nb::dict solve(amrex::MultiFab& rhs, amrex::MultiFab& sol)
    {
        resLogger_->clear(); // per-call history
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

        if (projectNullspace_)
        {
            // Singular system with the constant nullspace (e.g. fully-periodic
            // pure Poisson): make the rhs consistent by removing its mean, and
            // keep the initial guess in the mean-zero subspace so CG stays there.
            subtractMean(b_.get());
            subtractMean(x_.get());
        }

        solver_->apply(b_, x_);

        if (projectNullspace_)
        {
            // Pin the arbitrary constant: return the mean-zero representative
            // (also removes any roundoff drift out of the subspace).
            subtractMean(x_.get());
        }

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

        // Final 2-norm residual ||b - A x|| for reporting.
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

    PersistentSolver(std::shared_ptr<const gko::Executor> exec, gko::size_type n)
        : exec_(std::move(exec)), onDevice_(exec_->get_master().get() != exec_.get()), n_(n)
    {
        b_ = Dense::create(exec_, gko::dim<2> {n_, 1});
        x_ = Dense::create(exec_, gko::dim<2> {n_, 1});
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
        const std::string& precond
    )
        : PersistentSolver(
              makeExecutor(executor), static_cast<gko::size_type>(alpha->boxArray().numPts())
          )
    {
        const BcArray bcArr = parseBc(bc, geom, "FaceCoeffSolver");
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
        std::shared_ptr<const gko::LinOp> pc;
        if (precond == "gmg")
        {
            if (precond_mlmg != nullptr)
            {
                throw std::runtime_error(
                    "FaceCoeffSolver: precond='gmg' cannot be combined with precond_mlmg"
                );
            }
            pc = gko::share(GmgPrecond::create(
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
                bcArr,
                precond_cycles
            ));
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
        const std::string& precond
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
               const std::string& precond)
            {
                new (self) S(
                    executor, geom, &alpha, &ux, &lx, &uy, &ly, &uz, &lz, solver, max_iter, rtol,
                    atol, project_nullspace, precond_mlmg, precond_cycles, bc, precond
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
}
