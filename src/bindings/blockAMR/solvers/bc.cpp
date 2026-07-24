// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "bc.hpp"

#include <AMReX_Arena.H>

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
    const bool touches =
        low ? vbx.smallEnd(dir) == domain.smallEnd(dir) : vbx.bigEnd(dir) == domain.bigEnd(dir);
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

} // namespace blockamr::solvers
