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

#include <AMReX_Arena.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>

#include <ginkgo/ginkgo.hpp>

#include <algorithm>
#include <array>
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
        const amrex::MultiFab* lz
    )
        : gko::EnableLinOp<FaceCoeffOp>(exec, gko::dim<2> {n, n}), geom_(geom),
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
            amrex::Gpu::streamSynchronize();
            // Stencil is a free function: nvcc forbids an extended __device__
            // lambda inside a protected/private member.
            faceCoeffStencilDevice(*in_, *out_, *ux_, *lx_, *uy_, *ly_, *uz_, *lz_, *alpha_);
            gather_device(*out_, gko::as<Dense>(x)->get_values(), 1.0);
            amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
            return;
        }

        scatter(gko::as<Dense>(b)->get_const_values(), *in_);
        // Fill periodic + internal-box ghosts. Physical-boundary ghosts stay
        // whatever scatter left (untouched valid-only write); boundary faces
        // must carry a zero coefficient for those to be harmless.
        in_->FillBoundary(geom_.periodicity());
        amrex::Gpu::streamSynchronize();

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

// Build a Krylov solver over `op`, stopping on iteration count or the relative
// residual ||r|| <= rtol*||rhs|| (recomputed per solve, so one generate() is
// reused across right-hand sides).
std::shared_ptr<gko::LinOp> buildKrylov(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    int max_iter,
    double rtol
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
    if (solver == "cg")
    {
        return gko::solver::Cg<double>::build().with_criteria(criteria).on(exec)->generate(op);
    }
    if (solver == "bicgstab")
    {
        return gko::solver::Bicgstab<double>::build().with_criteria(criteria).on(exec)->generate(op
        );
    }
    if (solver == "gmres")
    {
        return gko::solver::Gmres<double>::build().with_criteria(criteria).on(exec)->generate(op);
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

        solver_->apply(b_, x_);

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
    void build(std::shared_ptr<gko::LinOp> op, const std::string& solver, int max_iter, double rtol)
    {
        op_ = std::move(op);
        solver_ = buildKrylov(solver, exec_, op_, max_iter, rtol);
        logger_ = gko::share(gko::log::Convergence<double>::create());
        solver_->add_logger(logger_);
    }

    std::shared_ptr<const gko::Executor> exec_;
    bool onDevice_;
    gko::size_type n_;
    std::shared_ptr<gko::LinOp> op_;
    std::unique_ptr<Dense> b_;
    std::unique_ptr<Dense> x_;
    std::shared_ptr<gko::LinOp> solver_;
    std::shared_ptr<gko::log::Convergence<double>> logger_;
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
        double rtol
    )
        : PersistentSolver(
              makeExecutor(executor), static_cast<gko::size_type>(alpha->boxArray().numPts())
          )
    {
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
            lz
        ));
        build(op, solver, max_iter, rtol);
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
        double rtol
    )
        : PersistentSolver(
              makeExecutor(executor), static_cast<gko::size_type>(alpha->boxArray().numPts())
          )
    {
        auto op = assembleFaceCoeffCsr(exec_, geom, *alpha, *ux, *lx, *uy, *ly, *uz, *lz);
        build(op, solver, max_iter, rtol);
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
               double rtol) {
                new (self)
                    S(executor, geom, &alpha, &ux, &lx, &uy, &ly, &uz, &lz, solver, max_iter, rtol);
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
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>(),
            nb::keep_alive<1, 4>(),
            nb::keep_alive<1, 5>(),
            nb::keep_alive<1, 6>(),
            nb::keep_alive<1, 7>(),
            nb::keep_alive<1, 8>()
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
            "solver, re-read each call so in-place updates take effect). Returns a\n"
            "dict with num_iters and res_norm."
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
            const double rhsNorm = rhs.norm2(0);
            const double stopTol = (rhsNorm > 0.0) ? rtol * rhsNorm : rtol;
            auto logger = gko::share(gko::log::Convergence<double>::create());
            auto solver = gko::solver::Cg<double>::build()
                              .with_criteria(
                                  gko::stop::Iteration::build().with_max_iters(
                                      static_cast<gko::size_type>(max_iter)
                                  ),
                                  gko::stop::ResidualNorm<double>::build()
                                      .with_baseline(gko::stop::mode::absolute)
                                      .with_reduction_factor(stopTol)
                              )
                              .on(exec)
                              ->generate(op);
            solver->add_logger(logger);
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
            return result;
        },
        nb::arg("lp"),
        nb::arg("sol"),
        nb::arg("rhs"),
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10,
        nb::arg("sign") = -1.0,
        nb::arg("executor") = "reference",
        "Matrix-free Ginkgo CG solve of the MLLinOp system L(sol) = rhs.\n\n"
        "sol's incoming values are the initial guess, and boundary data set\n"
        "via set_level_bc is honored (residual-correction solve). `sign` must\n"
        "make sign*L SPD: -1.0 (default) for MLPoisson (L = +laplacian,\n"
        "negative-definite); +1.0 for MLABecLaplacian (alpha*a*phi -\n"
        "beta*div(b grad phi), positive-definite). CG stops when\n"
        "||r_k|| <= rtol*||rhs||, so a warm start converges immediately.\n"
        "`executor` is 'reference' (CPU, default) or 'cuda' (GPU device 0). On\n"
        "'cuda' the entire solve runs on the device: the Krylov vector ops, the\n"
        "MLMG::apply mat-vec, and the vector<->MultiFab pack/unpack kernels all\n"
        "stay on the GPU, with no per-iteration host transfer. Returns a dict\n"
        "with num_iters and res_norm (2-norm of the homogeneous-system residual)."
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
