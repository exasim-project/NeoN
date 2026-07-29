// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/operators/laplacian.hpp"

#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "NeoN/blockAmr/core/parallelAlgorithms.hpp"

namespace blockamr::ops
{

namespace
{

// Explicit rather than left to AMREX_ASSERT, which is compiled out in a Release
// build: reading one field through an MFIter over a differently decomposed one
// would be a silent wrong answer.
void requireSameLayout(const amrex::MultiFab& mf, const amrex::MultiFab& like, const char* what)
{
    if (mf.boxArray() != like.boxArray() || mf.DistributionMap() != like.DistributionMap())
    {
        throw std::runtime_error(
            std::string("ops::Laplacian: ") + what
            + " has a different BoxArray/DistributionMapping than gamma"
        );
    }
}

/* @brief Accumulate the face coefficients of one direction, for a matrix that
 *        either has a separate low side (Asym) or does not.
 *
 * ONE body with a compile-time switch, not two hand-written copies. The two
 * instantiations are the SAME arithmetic in the SAME order -- they differ only in
 * whether the low-side coefficient lands in a second field -- and this operator's
 * arithmetic is pinned BITWISE by
 * test_laplacian_writes_the_boundary_face_coefficient. Two copies could drift
 * by a term, a sign or an association and still both compile; `if constexpr` over
 * one body makes that divergence unexpressible.
 *
 * Symmetry is decided ONCE, on the host, by the caller below. It used to be a
 * captured bool re-tested per cell -- a compile-time fact spelled as a runtime one.
 */
template<bool Asym>
void accumulateFaceCoefficients(
    const NeoN::Executor& exec,
    const amrex::MultiFab& g,
    amrex::MultiFab& upper,
    amrex::MultiFab* lower, // non-null iff Asym; never dereferenced otherwise
    int ex,
    int ey,
    int ez,
    bool periodicLo,
    bool periodicHi,
    int domLo,
    int domHi,
    amrex::Real invDx2
)
{
    for (amrex::MFIter mfi(upper); mfi.isValid(); ++mfi)
    {
        const amrex::Box& fbx = mfi.validbox();
        const auto G = g.const_array(mfi);
        const auto U = upper.array(mfi);
        // AMReX's own documented empty accessor (AMReX_Array4.H: "Default-construct
        // an empty accessor. The resulting accessor is invalid") -- constexpr,
        // host+device, p == nullptr. It replaces a dummy accessor that, when there
        // was no low side, was aliased onto U -- the very field this kernel
        // ACCUMULATES INTO. One lost guard there produced a plausible matrix with
        // every coefficient exactly 2x. Never read here: the write below is not
        // compiled at all when !Asym.
        amrex::Array4<amrex::Real> L {};
        if constexpr (Asym)
        {
            L = lower->array(mfi);
        }
        // EXPLICIT capture list, not `[=]`, and that is required rather than
        // stylistic: nvcc rejects an extended __device__ lambda that FIRST-captures
        // a variable inside an `if constexpr` block ("An extended __device__ lambda
        // cannot first-capture variable in constexpr-if context"), which is exactly
        // where `L` would first appear under a default capture. Naming it in the
        // capture-clause moves the capture out of the discarded branch. Everything
        // is by value; a __device__ lambda cannot capture by reference anyway.
        blockamr::parallelFor(
            exec,
            fbx,
            [G, U, L, ex, ey, ez, periodicLo, periodicHi, domLo, domHi, invDx2] AMREX_GPU_DEVICE(
                int i, int j, int k
            )
            {
                // Face f separates cell f-1 (low) from cell f (high) --
                // upper[d](f) is cell f-1's coefficient towards f, lower[d](f)
                // is cell f's towards f-1 (sparse/csr.cpp reads them at exactly
                // those offsets). One face value, both roles.
                const int f = (ex != 0) ? i : ((ey != 0) ? j : k);
                const bool atLo = !periodicLo && f == domLo;
                const bool atHi = !periodicHi && f == domHi + 1;
                // A NON-PERIODIC DOMAIN FACE CARRIES ITS REAL COEFFICIENT: the
                // diagonal half of the BC is the consumer's, per level, and every
                // consumer is MULTIPLICATIVE in this coefficient. DO NOT re-zero it
                // to fold the BC here -- an operator-side fold can only be right on
                // the finest level (folding (sign-1)*aF into alpha cost 12/13/14
                // Dirichlet iterations against 8/8/8). laplacian.hpp carries the
                // full contract, the measurements and the guarding test.
                //
                // gamma on such a face is the boundary cell's own: the ghost beyond
                // it is never filled (assemble() below), so reading it would read
                // whatever the arena recycled. Written as the mean of the interior
                // value with itself so one expression covers every face.
                const amrex::Real gLo = atLo ? G(i, j, k) : G(i - ex, j - ey, k - ez);
                const amrex::Real gHi = atHi ? G(i - ex, j - ey, k - ez) : G(i, j, k);
                const amrex::Real coef = -0.5 * (gLo + gHi) * invDx2;
                // Accumulate: several operators may share one system.
                U(i, j, k) += coef;
                if constexpr (Asym)
                {
                    L(i, j, k) += coef;
                }
            }
        );
    }
}

/* @brief Fold the INHOMOGENEOUS boundary datum of one non-periodic domain side
 *        into the rhs.
 *
 * The diagonal half of the boundary condition is NOT here -- see the note in
 * accumulateFaceCoefficients above: it belongs to whoever applies the matrix, per
 * level. This half cannot, because the consumers reachable through la:: never see
 * a datum: MFFaceCoeffs::op() hands FaceCoeffOp a null bcData
 * (faceCoeffMatrix.hpp) so FaceCoeffOpT::applyBcOffset is unreachable from
 * la::Solver, and assembleFaceCoeffCsr takes no datum at all. So the affine
 * constant c0 = aF*scale*g reaches the system only from here, once, at assembly.
 */
void foldBoundaryDatum(
    const NeoN::Executor& exec,
    const amrex::MultiFab& g,
    amrex::MultiFab& rhs,
    const amrex::MultiFab& bcData,
    const amrex::Box& slab,
    int ox,
    int oy,
    int oz,
    amrex::Real scale,
    amrex::Real invDx2
)
{
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = mfi.validbox() & slab;
        if (!bx.ok())
        {
            continue; // this box does not touch that domain face
        }
        const auto G = g.const_array(mfi);
        const auto R = rhs.array(mfi);
        const auto BD = bcData.const_array(mfi);
        // Explicit capture list to match the kernel above; everything is by value,
        // as a __device__ lambda cannot capture by reference anyway.
        blockamr::parallelFor(
            exec,
            bx,
            [G, R, BD, ox, oy, oz, scale, invDx2] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                // The same value the face kernel above writes onto this cell's
                // domain face, recomputed rather than read back: the face field is
                // ACCUMULATED into and may already carry another operator's
                // contribution, which is not this operator's datum to scale.
                const amrex::Real gC = G(i, j, k);
                const amrex::Real coef = -0.5 * (gC + gC) * invDx2;
                R(i, j, k) -= coef * scale * BD(i + ox, j + oy, k + oz);
            }
        );
    }
}

} // namespace

Laplacian::Laplacian(const amrex::MultiFab& gamma, la::BcArray bc, const amrex::MultiFab* bcData)
    : gamma_(&gamma), bc_(bc), bcData_(bcData)
{}

void Laplacian::assemble(la::Coefficients c) const
{
    // One-ghost staging copy of gamma. The coefficient on a face is the mean of
    // the two cells the face separates, and at a box edge the second cell lives in
    // a ghost: FillBoundary supplies the internal and periodic ones. On a
    // NON-periodic domain face there is no second cell at all, and the kernels
    // below use the single interior cell's gamma instead of reading the (unfilled)
    // ghost. That is an interpolation choice about a coefficient FIELD, separate
    // from the boundary condition applied to the solution.
    amrex::MultiFab g(gamma_->boxArray(), gamma_->DistributionMap(), 1, 1);
    amrex::MultiFab::Copy(g, *gamma_, 0, 0, 1, 0);
    c.mesh.fillHalo(g);

    const amrex::Box dom = c.mesh.geom.Domain();
    const auto dx = c.mesh.dx();

    // The datum fold writes a cell-centred field; validate it once, up front, and
    // only when a side actually reads a datum. Nothing cell-centred is written
    // without one -- the homogeneous boundary condition lives entirely on the face
    // coefficients.
    // `diag` and `rhs` are non-nullable handles (coefficients.hpp), so there is
    // nothing left to check for their presence -- the type says it.
    const bool anyPhysBc = std::any_of(bc_.begin(), bc_.end(), [](int b) { return b != 0; });
    if (anyPhysBc && bcData_ != nullptr)
    {
        requireSameLayout(*c.rhs, g, "the system's rhs");
        // `diag` is the layout checkBcData compares the datum against, so it has to
        // be the layout the fold below reads the datum THROUGH -- otherwise a
        // validated datum can still be indexed by the wrong MFIter.
        requireSameLayout(*c.diag, g, "the matrix's diagonal source");
        la::checkBcData(*bcData_, *c.diag, bc_, "ops::Laplacian");
    }
    else if (bcData_ != nullptr)
    {
        // Refused rather than ignored, exactly as FaceCoeffSolver refuses it:
        // a datum no side would ever read reads as a solver bug, not a
        // configuration one.
        la::checkBcData(*bcData_, *gamma_, bc_, "ops::Laplacian");
    }

    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab& upper = c.upper[d];

        const amrex::IntVect dv = amrex::IntVect::TheDimensionVector(d);
        if (upper.DistributionMap() != g.DistributionMap()
            || upper.boxArray() != amrex::convert(g.boxArray(), dv))
        {
            // Explicit rather than left to AMREX_ASSERT, which is compiled out in
            // a Release build: reading gamma through an MFIter over a differently
            // decomposed face field would be a silent wrong answer.
            throw std::runtime_error(
                "ops::Laplacian: gamma's BoxArray/DistributionMapping does not match the "
                "matrix's face coefficients"
            );
        }

        // Per SIDE, not per direction: BcArray is (xlo, xhi, ylo, yhi, zlo, zhi)
        // and 0 means periodic.
        const bool periodicLo = bc_[static_cast<std::size_t>(2 * d)] == 0;
        const bool periodicHi = bc_[static_cast<std::size_t>(2 * d + 1)] == 0;
        const int domLo = dom.smallEnd(d);
        const int domHi = dom.bigEnd(d);
        const int ex = (d == 0) ? 1 : 0;
        const int ey = (d == 1) ? 1 : 0;
        const int ez = (d == 2) ? 1 : 0;
        const amrex::Real invDx2 = 1.0 / (dx[d] * dx[d]);

        // Symmetry dispatched HERE, on the host, once per direction. A nullopt
        // `lower` IS the interface saying "there is no low side to write"
        // (coefficients.hpp): for a symmetric format lower[d] ALIASES upper[d] in
        // storage, so writing both would double every coefficient.
        if (c.lower.has_value())
        {
            accumulateFaceCoefficients<true>(
                c.exec,
                g,
                upper,
                &(*c.lower)[d],
                ex,
                ey,
                ez,
                periodicLo,
                periodicHi,
                domLo,
                domHi,
                invDx2
            );
        }
        else
        {
            accumulateFaceCoefficients<false>(
                c.exec, g, upper, nullptr, ex, ey, ez, periodicLo, periodicHi, domLo, domHi, invDx2
            );
        }

        // A homogeneous boundary condition is complete once the face coefficients
        // above are written -- the diagonal half is the consumers', per level (see
        // accumulateFaceCoefficients). Only the inhomogeneous datum still has to be
        // folded here, because no la:: consumer ever receives one.
        if (bcData_ == nullptr)
        {
            continue;
        }

        // Over the boundary CELLS rather than the boundary faces: a face kernel
        // would have two threads adding to one cell on a domain one cell thick, and
        // the rhs is cell-centred anyway. Low and high are separate launches for
        // the same reason.
        for (int s = 2 * d; s <= 2 * d + 1; ++s)
        {
            const int kind = bc_[static_cast<std::size_t>(s)];
            if (kind == 0)
            {
                continue;
            }
            const bool low = (s % 2) == 0;
            // core/bc.hpp's fillDomainBcGhostsInhom*: scale 2 for dirichlet (g is
            // u ON the face), scale dx for neumann (g is du/dn outward, and the
            // two flips of a low side cancel, so the scale carries no side sign).
            const amrex::Real scale = (kind == 1) ? 2.0 : dx[d];
            const int cell = low ? domLo : domHi;
            // From the boundary cell to the ghost cell holding its datum.
            const int step = low ? -1 : 1;
            const int ox = ex * step;
            const int oy = ey * step;
            const int oz = ez * step;
            amrex::Box slab = dom;
            slab.setSmall(d, cell);
            slab.setBig(d, cell);

            foldBoundaryDatum(c.exec, g, *c.rhs, *bcData_, slab, ox, oy, oz, scale, invDx2);
        }
    }
    amrex::Gpu::streamSynchronize();
}

} // namespace blockamr::ops
