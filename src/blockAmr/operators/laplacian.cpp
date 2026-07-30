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

// Explicit rather than left to AMREX_ASSERT, compiled out in Release: reading one field
// through an MFIter over a differently decomposed one is a silent wrong answer.
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

/* @brief Accumulate one direction's face coefficients, with (Asym) or without a separate
 *        low side. ONE body with a compile-time switch, not two copies that could drift by
 *        a sign and both compile; the arithmetic is pinned bitwise by the tripwire test.
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
        // AMReX's documented empty accessor: constexpr, host+device, p == nullptr. It
        // replaces a dummy that was aliased onto U -- the field this kernel ACCUMULATES
        // INTO -- where one lost guard gave a plausible matrix with every coefficient 2x.
        amrex::Array4<amrex::Real> L {};
        if constexpr (Asym)
        {
            L = lower->array(mfi);
        }
        // EXPLICIT capture list, required not stylistic: nvcc rejects an extended
        // __device__ lambda that FIRST-captures a variable inside an `if constexpr` block,
        // which is exactly where `L` would appear under a default capture.
        blockamr::parallelFor(
            exec,
            fbx,
            [G, U, L, ex, ey, ez, periodicLo, periodicHi, domLo, domHi, invDx2] AMREX_GPU_DEVICE(
                int i, int j, int k
            )
            {
                // Face f separates cell f-1 from cell f: upper[d](f) is f-1's coefficient
                // towards f, lower[d](f) is f's towards f-1. One face value, both roles.
                const int f = (ex != 0) ? i : ((ey != 0) ? j : k);
                const bool atLo = !periodicLo && f == domLo;
                const bool atHi = !periodicHi && f == domHi + 1;
                // A non-periodic domain face keeps its REAL coefficient: DO NOT re-zero it
                // to fold the BC here (laplacian.hpp). gamma on such a face is the
                // boundary cell's own -- the ghost beyond it is never filled.
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

/* @brief Fold the INHOMOGENEOUS boundary datum of one non-periodic domain side into the
 *        rhs. The diagonal half is not here (see above); this half cannot be, because no
 *        la:: consumer receives a datum, so c0 = aF*scale*g arrives only from here.
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
        // Explicit capture list to match the kernel above; everything is by value.
        blockamr::parallelFor(
            exec,
            bx,
            [G, R, BD, ox, oy, oz, scale, invDx2] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                // Recomputed rather than read back from the face field: that field is
                // ACCUMULATED into and may carry another operator's contribution.
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
    // One-ghost staging copy of gamma: a face coefficient is the mean of the two cells the
    // face separates, and at a box edge the second lives in a ghost FillBoundary supplies.
    // A non-periodic domain face has no second cell, so the kernels use the interior one.
    amrex::MultiFab g(gamma_->boxArray(), gamma_->DistributionMap(), 1, 1);
    amrex::MultiFab::Copy(g, *gamma_, 0, 0, 1, 0);
    c.mesh.fillHalo(g);

    const amrex::Box dom = c.mesh.geom.Domain();
    const auto dx = c.mesh.dx();

    // Validate the datum carrier once, up front, and only when a side actually reads one.
    // `diag` and `rhs` are non-nullable handles, so their presence needs no check.
    const bool anyPhysBc = std::any_of(bc_.begin(), bc_.end(), [](int b) { return b != 0; });
    if (anyPhysBc && bcData_ != nullptr)
    {
        requireSameLayout(*c.rhs, g, "the system's rhs");
        // `diag` is the layout checkBcData compares against, so it must be the layout the
        // fold below reads the datum THROUGH.
        requireSameLayout(*c.diag, g, "the matrix's diagonal source");
        la::checkBcData(*bcData_, *c.diag, bc_, "ops::Laplacian");
    }
    else if (bcData_ != nullptr)
    {
        // Refused rather than ignored, as FaceCoeffSolver refuses it: a datum no side
        // reads is a solver bug, not a configuration one.
        la::checkBcData(*bcData_, *gamma_, bc_, "ops::Laplacian");
    }

    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab& upper = c.upper[d];

        const amrex::IntVect dv = amrex::IntVect::TheDimensionVector(d);
        if (upper.DistributionMap() != g.DistributionMap()
            || upper.boxArray() != amrex::convert(g.boxArray(), dv))
        {
            // Explicit rather than left to AMREX_ASSERT, compiled out in Release: a
            // silent wrong answer otherwise.
            throw std::runtime_error(
                "ops::Laplacian: gamma's BoxArray/DistributionMapping does not match the "
                "matrix's face coefficients"
            );
        }

        // Per SIDE, not per direction: BcArray is (xlo, xhi, ylo, yhi, zlo, zhi).
        const bool periodicLo = bc_[static_cast<std::size_t>(2 * d)] == 0;
        const bool periodicHi = bc_[static_cast<std::size_t>(2 * d + 1)] == 0;
        const int domLo = dom.smallEnd(d);
        const int domHi = dom.bigEnd(d);
        const int ex = (d == 0) ? 1 : 0;
        const int ey = (d == 1) ? 1 : 0;
        const int ez = (d == 2) ? 1 : 0;
        const amrex::Real invDx2 = 1.0 / (dx[d] * dx[d]);

        // Symmetry dispatched HERE, on the host. A nullopt `lower` IS the interface saying
        // there is no low side: for a symmetric format lower[d] ALIASES upper[d].
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

        // A homogeneous BC is complete once the face coefficients are written; only the
        // inhomogeneous datum is folded here, since no la:: consumer receives one.
        if (bcData_ == nullptr)
        {
            continue;
        }

        // Over the boundary CELLS, not faces: a face kernel would have two threads adding
        // to one cell on a domain one cell thick. Low and high are separate launches.
        for (int s = 2 * d; s <= 2 * d + 1; ++s)
        {
            const int kind = bc_[static_cast<std::size_t>(s)];
            if (kind == 0)
            {
                continue;
            }
            const bool low = (s % 2) == 0;
            // core/bc.hpp's inhom fill: scale 2 for dirichlet (g is u ON the face), dx for
            // neumann (g is du/dn outward; a low side's two flips cancel).
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
