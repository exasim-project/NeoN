// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/operators/laplacian.hpp"

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/core/fieldLevel.hpp"
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

// Everything one direction's stencil needs, as one trivially-copyable device capture.
struct Axis
{
    int dir;
    int ex, ey, ez;
    bool periodicLo, periodicHi;
    int domLo, domHi;
    amrex::Real invDx2;
};

// Build direction d's Axis; `bc` is indexed per SIDE, as (xlo, xhi, ylo, yhi, zlo, zhi).
Axis makeAxis(int d, const amrex::Box& dom, const la::BcArray& bc, amrex::Real dx)
{
    Axis a {};
    a.dir = d;
    a.ex = (d == 0) ? 1 : 0;
    a.ey = (d == 1) ? 1 : 0;
    a.ez = (d == 2) ? 1 : 0;
    a.periodicLo = bc[static_cast<std::size_t>(2 * d)] == 0;
    a.periodicHi = bc[static_cast<std::size_t>(2 * d + 1)] == 0;
    a.domLo = dom.smallEnd(d);
    a.domHi = dom.bigEnd(d);
    a.invDx2 = 1.0 / (dx * dx);
    return a;
}

// Explicit rather than left to AMREX_ASSERT, compiled out in Release: a silent wrong answer.
void requireFaceLayout(const amrex::MultiFab& upper, const amrex::MultiFab& g, int d)
{
    const amrex::IntVect dv = amrex::IntVect::TheDimensionVector(d);
    if (upper.DistributionMap() != g.DistributionMap()
        || upper.boxArray() != amrex::convert(g.boxArray(), dv))
    {
        throw std::runtime_error(
            "ops::Laplacian: gamma's BoxArray/DistributionMapping does not match the "
            "matrix's face coefficients"
        );
    }
}

// Validate the datum carrier once, up front, and only when a side actually reads one.
void checkDatumCarrier(
    la::LinearSystem& sys,
    const amrex::MultiFab& g,
    const amrex::MultiFab& gamma,
    const amrex::MultiFab* bcData
)
{
    if (bcData == nullptr)
    {
        return;
    }
    auto& m = sys.matrix();
    const bool anyPhysBc = std::any_of(m.bc.begin(), m.bc.end(), [](int b) { return b != 0; });
    if (!anyPhysBc)
    {
        // Refused rather than ignored, as FaceCoeffSolver refuses it: a datum no side
        // reads is a solver bug, not a configuration one.
        la::checkBcData(*bcData, gamma, m.bc, "ops::Laplacian");
        return;
    }
    requireSameLayout(sys.rhs(), g, "the system's rhs");
    // `alpha` is the layout checkBcData compares against, so it must be the layout the
    // fold below reads the datum THROUGH.
    requireSameLayout(*m.alpha, g, "the matrix's diagonal source");
    la::checkBcData(*bcData, *m.alpha, m.bc, "ops::Laplacian");
}

// The low side to accumulate into, or nothing when the matrix is symmetric.
amrex::Array4<amrex::Real>
lowSideView(std::optional<FaceFieldLevel>& lower, int d, const amrex::MFIter& mfi)
{
    if (!lower.has_value())
    {
        // AMReX's documented empty accessor: constexpr, host+device, p == nullptr. It replaces
        // a dummy that was aliased onto upper -- the field the kernel ACCUMULATES INTO -- where
        // one lost guard gave a plausible matrix with every coefficient 2x.
        return {};
    }
    return (*lower)[d].array(mfi);
}

// One non-periodic side: the slab of boundary CELLS and the offset to each one's datum ghost.
struct BoundarySide
{
    Axis axis;
    amrex::Box slab;
    int ox, oy, oz;
    amrex::Real scale;
};

// Build the low or high side of `axis`; `scale` is core/bc.hpp's inhomogeneous fill factor.
BoundarySide makeBoundarySide(const Axis& axis, const amrex::Box& dom, bool low, amrex::Real scale)
{
    const int cell = low ? axis.domLo : axis.domHi;
    const int step = low ? -1 : 1;
    amrex::Box slab = dom;
    slab.setSmall(axis.dir, cell);
    slab.setBig(axis.dir, cell);
    return BoundarySide {axis, slab, axis.ex * step, axis.ey * step, axis.ez * step, scale};
}

// Fold one side's INHOMOGENEOUS datum into the rhs; the diagonal half is not here (laplacian.hpp).
void foldBoundaryDatum(
    const NeoN::Executor& exec,
    const amrex::MultiFab& g,
    amrex::MultiFab& rhs,
    const amrex::MultiFab& bcData,
    BoundarySide side
)
{
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = mfi.validbox() & side.slab;
        if (!bx.ok())
        {
            continue; // this box does not touch that domain face
        }
        const auto G = g.const_array(mfi);
        const auto R = rhs.array(mfi);
        const auto BD = bcData.const_array(mfi);
        blockamr::parallelFor(
            exec,
            bx,
            BLOCKAMR_LAMBDA(int i, int j, int k) {
                // Recomputed rather than read back from the face field: that field is
                // ACCUMULATED into and may carry another operator's contribution.
                const amrex::Real gC = G(i, j, k);
                const amrex::Real coef = -0.5 * (gC + gC) * side.axis.invDx2;
                R(i, j, k) -= coef * side.scale * BD(i + side.ox, j + side.oy, k + side.oz);
            }
        );
    }
}

// Fold both of direction `axis`'s sides; a periodic or homogeneous one has nothing to fold.
void foldBoundaryData(
    la::LinearSystem& sys, const amrex::MultiFab& g, const amrex::MultiFab* bcData, const Axis& axis
)
{
    if (bcData == nullptr)
    {
        return;
    }
    auto& m = sys.matrix();
    const amrex::Box dom = m.mesh.geom.Domain();
    const amrex::Real dx = m.mesh.dx()[axis.dir];
    // Over the boundary CELLS, not faces: a face kernel would have two threads adding to one
    // cell on a domain one cell thick. Low and high are separate launches.
    for (int s = 2 * axis.dir; s <= 2 * axis.dir + 1; ++s)
    {
        const int kind = m.bc[static_cast<std::size_t>(s)];
        if (kind == 0)
        {
            continue;
        }
        // core/bc.hpp's inhom fill: scale 2 for dirichlet (g is u ON the face), dx for
        // neumann (g is du/dn outward; a low side's two flips cancel).
        const amrex::Real scale = (kind == 1) ? 2.0 : dx;
        const BoundarySide side = makeBoundarySide(axis, dom, (s % 2) == 0, scale);
        foldBoundaryDatum(m.exec, g, sys.rhs(), *bcData, side);
    }
}

} // namespace

Laplacian::Laplacian(const amrex::MultiFab& gamma, const amrex::MultiFab* bcData)
    : gamma_(&gamma), bcData_(bcData)
{}

void Laplacian::assemble(la::LinearSystem& sys) const
{
    auto& m = sys.matrix();
    // One ghost: at a box edge the face's second cell is one fillHalo supplies.
    amrex::MultiFab g(gamma_->boxArray(), gamma_->DistributionMap(), 1, 1);
    amrex::MultiFab::Copy(g, *gamma_, 0, 0, 1, 0);
    m.mesh.fillHalo(g);
    checkDatumCarrier(sys, g, *gamma_, bcData_);

    const amrex::Box dom = m.mesh.geom.Domain();
    const auto dx = m.mesh.dx();
    const bool asym = m.lower.has_value();
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab& upper = m.upper[d];
        requireFaceLayout(upper, g, d);
        const Axis ax = makeAxis(d, dom, m.bc, dx[d]);
        for (amrex::MFIter mfi(upper); mfi.isValid(); ++mfi)
        {
            const auto G = g.const_array(mfi);
            const auto U = upper.array(mfi);
            const auto L = lowSideView(m.lower, d, mfi);
            // BLOCKAMR_LAMBDA is [=]: every name the body reads must be a local, never a
            // member, which would capture `this` and deref a host pointer on the device.
            blockamr::parallelFor(
                m.exec,
                mfi.validbox(),
                BLOCKAMR_LAMBDA(int i, int j, int k) {
                    // Face f separates cell f-1 from cell f: upper[d](f) is f-1's coefficient
                    // towards f, lower[d](f) is f's towards f-1. One face value, both roles.
                    const int f = (ax.ex != 0) ? i : ((ax.ey != 0) ? j : k);
                    const bool atLo = !ax.periodicLo && f == ax.domLo;
                    const bool atHi = !ax.periodicHi && f == ax.domHi + 1;
                    const amrex::Real gLo = atLo ? G(i, j, k) : G(i - ax.ex, j - ax.ey, k - ax.ez);
                    const amrex::Real gHi = atHi ? G(i - ax.ex, j - ax.ey, k - ax.ez) : G(i, j, k);
                    const amrex::Real coef = -0.5 * (gLo + gHi) * ax.invDx2;
                    U(i, j, k) += coef;
                    if (asym)
                    {
                        L(i, j, k) += coef;
                    }
                }
            );
        }
        foldBoundaryData(sys, g, bcData_, ax);
    }
    amrex::Gpu::streamSynchronize();
}

// The operator holds NEITHER a Geometry NOR the domain BCs any more -- both are read off the
// system's matrix -- so neither of the two older spellings compiles. Asserted in the shipped
// object library rather than under test/, because blockAmr has no C++ test target; these two
// outlived linearAlgebra/coefficientsConcepts.cpp, whose subject (IsMatrix/IsOperator/
// Coefficients) no longer exists.
static_assert(!std::is_constructible_v<
              Laplacian,
              const amrex::MultiFab&,
              amrex::Geometry,
              la::BcArray>);
static_assert(!std::is_constructible_v<Laplacian, const amrex::MultiFab&, la::BcArray>);

} // namespace blockamr::ops
