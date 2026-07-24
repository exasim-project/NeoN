// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "face_coeff_op.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFabUtil.H>

#include <algorithm>

#include "profiling.hpp"
#include "transfer.hpp"

namespace blockamr::solvers
{

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

FaceCoeffOp::FaceCoeffOp(std::shared_ptr<const gko::Executor> exec)
    : AmrexLinOpBase<FaceCoeffOp>(exec)
{}

FaceCoeffOp::FaceCoeffOp(
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
    BcArray bc
)
    : AmrexLinOpBase<FaceCoeffOp>(exec, gko::dim<2> {n, n}), geom_(geom), bc_(bc),
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

void FaceCoeffOp::apply_impl(const gko::LinOp* b, gko::LinOp* x) const
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

} // namespace blockamr::solvers
