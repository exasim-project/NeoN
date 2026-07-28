// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/core/bc.hpp"

#include <AMReX_Arena.H>

#include <algorithm>
#include <stdexcept>

namespace blockamr::solvers
{

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

BcArray
parseBc(const std::vector<std::string>& bc, const amrex::Geometry& geom, const std::string& who)
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

void checkBcData(
    const amrex::MultiFab& bcdata,
    const amrex::MultiFab& like,
    const BcArray& bc,
    const std::string& who
)
{
    if (bcdata.boxArray() != like.boxArray() || bcdata.DistributionMap() != like.DistributionMap())
    {
        throw std::runtime_error(
            who + ": bc_data must share the BoxArray and DistributionMapping of alpha"
        );
    }
    if (bcdata.nGrow() < 1)
    {
        throw std::runtime_error(
            who
            + ": bc_data needs at least 1 ghost cell — the boundary datum lives in the "
              "ghost layer (MLMG's set_level_bc contract)"
        );
    }
    if (std::none_of(bc.begin(), bc.end(), [](int b) { return b != 0; }))
    {
        throw std::runtime_error(
            who + ": bc_data was given but every side is 'periodic', so nothing would read it"
        );
    }
}

template<class V>
void scatterShellDevice(const V* vec, amrex::MultiFab& mf)
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
                    a(i, j, k) = static_cast<amrex::Real>(vec[idx]);
                }
            }
        );
        off += vbx.numPts();
    }
}

// The two flat-vector value types the Krylov paths use: double for the fp64
// solvers, float for the mixed-precision inner solve.
template void scatterShellDevice<double>(const double*, amrex::MultiFab&);
template void scatterShellDevice<float>(const float*, amrex::MultiFab&);

} // namespace blockamr::solvers
