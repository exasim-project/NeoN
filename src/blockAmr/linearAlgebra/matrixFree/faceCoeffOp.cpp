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
#include "NeoN/blockAmr/linearAlgebra/stencil.hpp"
#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"

namespace blockamr::la
{

// The six face views over one box, in the HIGH/LOW order loadFaceCoeffs expects.
static FaceCoeffArrays<amrex::Real>
faceArrays(const FaceFieldLevel& upper, const FaceFieldLevel& lower, const amrex::MFIter& mfi)
{
    return {
        upper[0].const_array(mfi),
        lower[0].const_array(mfi),
        upper[1].const_array(mfi),
        lower[1].const_array(mfi),
        upper[2].const_array(mfi),
        lower[2].const_array(mfi)
    };
}

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
        const auto faces = faceArrays(upper, lower, mfi);
        blockamr::parallelFor(
            exec,
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const auto c = loadFaceCoeffs<amrex::Real>(faces, i, j, k);
                dg(i, j, k) = stencilDiag(al(i, j, k), c);
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
    const FaceCoeffLevel& level
)
{
    long off = 0;
    for (amrex::MFIter mfi(in); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = in.const_array(mfi);
        const auto faces = faceArrays(level.upper, level.lower, mfi);
        const auto al = (*level.alpha).const_array(mfi);
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
                const auto c = loadFaceCoeffs<V>(faces, i, j, k);
                const V offd = stencilOffDiag(c, pE, pW, pN, pS, pT, pB);
                // PROTOTYPE (C1): centre term recomputed inline instead of read from a stored
                // diagonal; same association order as computeFaceCoeffDiag.
                const V diag = stencilDiag(static_cast<V>(al(i, j, k)), c);
                xo[idx] = diag * pC + offd;
            }
        );
        off += vbx.numPts();
    }
}

// Host twin of the fused stencil: `in` already carries its ghost layer, so every neighbour is
// read from it and the flat vectors are not touched. Always fp64 -- the constructor refuses an
// fp32 host operator.
static void
faceCoeffStencilHost(const amrex::MultiFab& in, amrex::MultiFab& out, const FaceCoeffLevel& level)
{
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = in.const_array(mfi);
        const auto o = out.array(mfi);
        const auto faces = faceArrays(level.upper, level.lower, mfi);
        const auto al = (*level.alpha).const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const auto c = loadFaceCoeffs<double>(faces, i, j, k);
                    const double off = stencilOffDiag(
                        c,
                        psi(i + 1, j, k),
                        psi(i - 1, j, k),
                        psi(i, j + 1, k),
                        psi(i, j - 1, k),
                        psi(i, j, k + 1),
                        psi(i, j, k - 1)
                    );
                    // PROTOTYPE (C1): inline centre term, same association order.
                    const double dgv = stencilDiag(al(i, j, k), c);
                    o(i, j, k) = dgv * psi(i, j, k) + off;
                }
            }
        }
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
    const FaceCoeffLevel& level,
    DomainBc bc,
    const CellFieldLevel& diag
)
    // The row count IS the level's global cell count -- what every caller used to compute and
    // hand back in (la::globalRows).
    : AmrexLinOpBase<FaceCoeffOpT<V>, V>(
        exec,
        gko::dim<2> {
            static_cast<gko::size_type>(level.mesh.ba.numPts()),
            static_cast<gko::size_type>(level.mesh.ba.numPts())
        }
    ),
      nexec_(nexec), bc_(bc.sides),
      hasPhysBc_(std::any_of(bc.sides.begin(), bc.sides.end(), [](int b) { return b != 0; })),
      onDevice_(exec->get_master().get() != exec.get()), level_(level)
{
    for (int d = 0; d < 3; ++d)
    {
        dx_[d] = level_.mesh.geom.CellSize(d);
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
    const MeshLevel& mesh = level_.mesh;
    if (onDevice_)
    {
        // level_ already references the caller's device fields; in_/out_ live in the device
        // arena.
        bcData_ = bc.data;
        in_ = std::make_shared<amrex::MultiFab>(mesh.ba, mesh.dm, 1, 1);
        out_ = std::make_shared<amrex::MultiFab>(mesh.ba, mesh.dm, 1, 0);
    }
    else
    {
        stagePinned(level, bc.data);
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

// Host (ReferenceExecutor) stencil: the loop reads pinned memory, so a caller's in-place write
// after construction is not observed.
template<class V>
void FaceCoeffOpT<V>::stagePinned(const FaceCoeffLevel& level, const amrex::MultiFab* bcData)
{
    level_.alpha = CellFieldLevel {pinnedCopy(*level.alpha)};
    level_.upper = FaceFieldLevel {
        {pinnedCopy(level.upper[0]), pinnedCopy(level.upper[1]), pinnedCopy(level.upper[2])}
    };
    level_.lower = FaceFieldLevel {
        {pinnedCopy(level.lower[0]), pinnedCopy(level.lower[1]), pinnedCopy(level.lower[2])}
    };
    // The device-arena original is dead weight once the pinned copy exists.
    diagOwned_.reset();
    if (bcData != nullptr)
    {
        bcDataOwned_ = pinnedCopy(*bcData);
        bcData_ = bcDataOwned_.get();
    }
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
        applyFused(b, x, inhom);
        return;
    }
    applyStaged(b, x, inhom);
}

template<class V>
void FaceCoeffOpT<V>::fillGhostsDevice(bool inhom) const
{
    level_.mesh.fillHalo(*in_);
    if (!hasPhysBc_)
    {
        return;
    }
    // Reflect-odd/even: this is where the homogeneous Dirichlet/Neumann BCs enter the
    // stencil, once per apply.
    if (inhom)
    {
        fillDomainBcGhostsInhomDevice(nexec_, *in_, *bcData_, level_.mesh.geom.Domain(), bc_, dx_);
    }
    else
    {
        fillDomainBcGhostsDevice(*in_, level_.mesh.geom.Domain(), bc_);
    }
}

template<class V>
void FaceCoeffOpT<V>::applyFused(const gko::LinOp* b, gko::LinOp* x, bool inhom) const
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
        fillGhostsDevice(inhom);
    }
    amrex::Gpu::streamSynchronize();
    {
        prof::Timer t("op.stencil");
        // A free function: nvcc forbids an extended __device__ lambda in a member.
        faceCoeffStencilFusedDevice(nexec_, bvals, xvals, *in_, level_);
    }
    {
        prof::Timer t("op.gather");
        amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
    }
}

template<class V>
void FaceCoeffOpT<V>::applyStaged(const gko::LinOp* b, gko::LinOp* x, bool inhom) const
{
    scatter(localValues<V>(b), *in_);
    // Periodic + internal-box ghosts; the reflect fill below sets physical-boundary ones. On
    // an all-periodic operator they stay whatever scatter left, so the boundary faces must
    // carry a zero coefficient for that to be harmless.
    level_.mesh.fillHalo(*in_);
    amrex::Gpu::streamSynchronize();
    if (hasPhysBc_)
    {
        if (inhom)
        {
            fillDomainBcGhostsInhomHost(*in_, *bcData_, level_.mesh.geom.Domain(), bc_, dx_);
        }
        else
        {
            fillDomainBcGhostsHost(*in_, level_.mesh.geom.Domain(), bc_);
        }
    }

    faceCoeffStencilHost(*in_, *out_, level_);
    gather(*out_, localValues<V>(x), 1.0);
}

// The two value types the Krylov paths use; FaceCoeffOpT<float> is device-only.
template class FaceCoeffOpT<double>;
template class FaceCoeffOpT<float>;

} // namespace blockamr::la
