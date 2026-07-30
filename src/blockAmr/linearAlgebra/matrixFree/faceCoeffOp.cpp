// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFabUtil.H>

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "NeoN/blockAmr/core/parallelAlgorithms.hpp"
#include "NeoN/blockAmr/core/profiling.hpp"
#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"

namespace blockamr::la
{

void computeFaceCoeffDiag(
    const NeoN::Executor& exec,
    CellFieldLevel diag,
    const CellFieldLevel& alpha,
    const FaceFieldLevel& upper,
    const FaceFieldLevel& lower
)
{
    amrex::MultiFab& dgf = *diag;
    for (amrex::MFIter mfi(dgf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto dg = dgf.array(mfi);
        const auto al = (*alpha).const_array(mfi);
        const auto ax = upper[0].const_array(mfi);
        const auto lxa = lower[0].const_array(mfi);
        const auto ay = upper[1].const_array(mfi);
        const auto lya = lower[1].const_array(mfi);
        const auto az = upper[2].const_array(mfi);
        const auto lza = lower[2].const_array(mfi);
        blockamr::parallelFor(
            exec,
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                // Same association order as the two stencils: aE=ux(high face), aW=lx(low), ...
                const amrex::Real aE = ax(i + 1, j, k);
                const amrex::Real aW = lxa(i, j, k);
                const amrex::Real aN = ay(i, j + 1, k);
                const amrex::Real aS = lya(i, j, k);
                const amrex::Real aT = az(i, j, k + 1);
                const amrex::Real aB = lza(i, j, k);
                dg(i, j, k) = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
            }
        );
    }
    amrex::Gpu::streamSynchronize();
}

// Fused matrix-free apply: the stencil reads the centre and interior neighbours from the flat
// Ginkgo input, the ghosted scratch `in` only where a neighbour leaves the valid box, and
// writes straight to the flat output -- bit-identical, and b/x must not alias.
template<class V>
void faceCoeffStencilFusedDevice(
    const NeoN::Executor& exec,
    const V* bvec,
    V* xvec,
    const amrex::MultiFab& in,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alphaMf
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
        const auto al = alphaMf.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        const int ni = vbx.length(0);
        const int nj = vbx.length(1);
        const long nij = static_cast<long>(ni) * nj;
        const long o = off;
        const V* b = bvec;
        V* xo = xvec;
        blockamr::parallelFor(
            exec,
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const long idx =
                    o + (static_cast<long>(k - lo.z) * nj + (j - lo.y)) * ni + (i - lo.x);
                // Locals stay V: at V = float this is an fp32 evaluation of the operator,
                // not an fp64 one with narrowed endpoints.
                const V pC = b[idx];
                const V pE = (i < hi.x) ? b[idx + 1] : static_cast<V>(psi(i + 1, j, k));
                const V pW = (i > lo.x) ? b[idx - 1] : static_cast<V>(psi(i - 1, j, k));
                const V pN = (j < hi.y) ? b[idx + ni] : static_cast<V>(psi(i, j + 1, k));
                const V pS = (j > lo.y) ? b[idx - ni] : static_cast<V>(psi(i, j - 1, k));
                const V pT = (k < hi.z) ? b[idx + nij] : static_cast<V>(psi(i, j, k + 1));
                const V pB = (k > lo.z) ? b[idx - nij] : static_cast<V>(psi(i, j, k - 1));
                const V aE = static_cast<V>(ax(i + 1, j, k));
                const V aW = static_cast<V>(lxa(i, j, k));
                const V aN = static_cast<V>(ay(i, j + 1, k));
                const V aS = static_cast<V>(lya(i, j, k));
                const V aT = static_cast<V>(az(i, j, k + 1));
                const V aB = static_cast<V>(lza(i, j, k));
                const V offd = aE * pE + aW * pW + aN * pN + aS * pS + aT * pT + aB * pB;
                // PROTOTYPE (C1): centre term recomputed inline instead of read from a stored
                // diagonal; same association order as computeFaceCoeffDiag.
                const V diag = static_cast<V>(al(i, j, k)) - (aE + aW + aN + aS + aT + aB);
                xo[idx] = diag * pC + offd;
            }
        );
        off += vbx.numPts();
    }
}

template<class V>
FaceCoeffOpT<V>::FaceCoeffOpT(std::shared_ptr<const gko::Executor> exec)
    : AmrexLinOpBase<FaceCoeffOpT<V>, V>(exec)
{}

template<class V>
FaceCoeffOpT<V>::FaceCoeffOpT(
    std::shared_ptr<const gko::Executor> exec,
    const NeoN::Executor& nexec,
    const MeshLevel& mesh,
    gko::size_type n,
    const CellFieldLevel& alpha,
    const FaceFieldLevel& upper,
    const FaceFieldLevel& lower,
    BcArray bc,
    const amrex::MultiFab* bcData,
    const CellFieldLevel& diag
)
    : AmrexLinOpBase<FaceCoeffOpT<V>, V>(exec, gko::dim<2> {n, n}), geom_(mesh.geom), nexec_(nexec),
      bc_(bc), hasPhysBc_(std::any_of(bc.begin(), bc.end(), [](int b) { return b != 0; })),
      onDevice_(exec->get_master().get() != exec.get())
{
    for (int d = 0; d < 3; ++d)
    {
        dx_[d] = geom_.CellSize(d);
    }
    // Only the device stencil is instantiated in V; the host loop computes in double, so an
    // fp32 host build is refused rather than silently run at fp64.
    if (!onDevice_ && !std::is_same_v<V, double>)
    {
        throw std::runtime_error(
            "FaceCoeffOp: the reduced-precision operator is a device path; use executor='cuda'"
        );
    }
    // PROTOTYPE (C1): no stored diagonal at all -- the stencils recompute alpha - sum(faces)
    // inline, so alpha is what they read.
    (void)diag;
    const amrex::MultiFab* diagField = &(*alpha);
    if (onDevice_)
    {
        // Reference the caller's device fields directly; in_/out_ live in the device arena.
        diag_ = diagField;
        ux_ = &upper[0];
        lx_ = &lower[0];
        uy_ = &upper[1];
        ly_ = &lower[1];
        uz_ = &upper[2];
        lz_ = &lower[2];
        bcData_ = bcData;
        in_ = std::make_shared<amrex::MultiFab>(mesh.ba, mesh.dm, 1, 1);
        out_ = std::make_shared<amrex::MultiFab>(mesh.ba, mesh.dm, 1, 0);
    }
    else
    {
        // Host (ReferenceExecutor) stencil: stage the coefficients to pinned memory once and
        // read those. Under PROTOTYPE (C1) diagField is alpha, one of them.
        owned_ = {
            pinnedCopy(*diagField),
            pinnedCopy(upper[0]),
            pinnedCopy(lower[0]),
            pinnedCopy(upper[1]),
            pinnedCopy(lower[1]),
            pinnedCopy(upper[2]),
            pinnedCopy(lower[2])
        };
        diag_ = owned_[0].get();
        ux_ = owned_[1].get();
        lx_ = owned_[2].get();
        uy_ = owned_[3].get();
        ly_ = owned_[4].get();
        uz_ = owned_[5].get();
        lz_ = owned_[6].get();
        // The device-arena original is dead weight once the pinned copy exists.
        diagOwned_.reset();
        if (bcData != nullptr)
        {
            owned_.push_back(pinnedCopy(*bcData));
            bcData_ = owned_.back().get();
        }
        in_ = std::make_shared<amrex::MultiFab>(
            mesh.ba, mesh.dm, 1, 1, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
        );
        out_ = std::make_shared<amrex::MultiFab>(
            mesh.ba, mesh.dm, 1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
        );
    }
    in_->setVal(0.0);
    out_->setVal(0.0);
}

template<class V>
void FaceCoeffOpT<V>::apply_impl(const gko::LinOp* b, gko::LinOp* x) const
{
    // The operator Ginkgo sees is always the LINEAR one: reflecting domain-BC ghosts, no
    // bcData. The inhomogeneous fill is reached only through applyBcOffset.
    applyWith(b, x, false);
}

template<class V>
void FaceCoeffOpT<V>::applyBcOffset(const gko::LinOp* zero, gko::LinOp* out) const
{
    if (bcData_ == nullptr)
    {
        throw std::runtime_error("FaceCoeffOp: applyBcOffset without bc_data");
    }
    applyWith(zero, out, true);
}

template<class V>
void FaceCoeffOpT<V>::applyWith(const gko::LinOp* b, gko::LinOp* x, bool inhom) const
{
    if (onDevice_)
    {
        prof::Timer tAll("op.apply");
        {
            prof::Timer t("op.sync_gko");
            this->get_executor()->synchronize(); // b written by Ginkgo
        }
        const V* bvals = localValues<V>(b);
        V* xvals = localValues<V>(x);
        {
            // Only the ghost-adjacent shell needs to reach the MultiFab; the stencil reads
            // the interior straight from the flat vector.
            prof::Timer t("op.scatter");
            scatterShellDevice(nexec_, bvals, *in_);
        }
        {
            prof::Timer t("op.fill");
            in_->FillBoundary(geom_.periodicity());
            if (hasPhysBc_)
            {
                // Reflect-odd/even: this is where the homogeneous Dirichlet/Neumann BCs
                // enter the stencil, once per apply.
                if (inhom)
                {
                    fillDomainBcGhostsInhomDevice(nexec_, *in_, *bcData_, geom_.Domain(), bc_, dx_);
                }
                else
                {
                    fillDomainBcGhostsDevice(*in_, geom_.Domain(), bc_);
                }
            }
        }
        amrex::Gpu::streamSynchronize();
        {
            prof::Timer t("op.stencil");
            // A free function: nvcc forbids an extended __device__ lambda in a member.
            faceCoeffStencilFusedDevice(
                nexec_, bvals, xvals, *in_, *ux_, *lx_, *uy_, *ly_, *uz_, *lz_, *diag_
            );
        }
        {
            prof::Timer t("op.gather");
            amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
        }
        return;
    }

    scatter(localValues<V>(b), *in_);
    // Periodic + internal-box ghosts; the reflect fill below sets physical-boundary ones. On
    // an all-periodic operator they stay whatever scatter left, so the boundary faces must
    // carry a zero coefficient for that to be harmless.
    in_->FillBoundary(geom_.periodicity());
    amrex::Gpu::streamSynchronize();
    if (hasPhysBc_)
    {
        if (inhom)
        {
            fillDomainBcGhostsInhomHost(*in_, *bcData_, geom_.Domain(), bc_, dx_);
        }
        else
        {
            fillDomainBcGhostsHost(*in_, geom_.Domain(), bc_);
        }
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
        const auto dg = diag_->const_array(mfi);
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
                    // PROTOTYPE (C1): inline centre term, same association order.
                    const double dgv = dg(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    o(i, j, k) = dgv * psi(i, j, k) + off;
                }
            }
        }
    }
    gather(*out_, localValues<V>(x), 1.0);
}

// The two value types the Krylov paths use; FaceCoeffOpT<float> is device-only.
template class FaceCoeffOpT<double>;
template class FaceCoeffOpT<float>;

} // namespace blockamr::la
