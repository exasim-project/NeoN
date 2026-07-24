// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "csr.hpp"

#include "bc.hpp"

#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blockamr::solvers
{

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

} // namespace blockamr::solvers
