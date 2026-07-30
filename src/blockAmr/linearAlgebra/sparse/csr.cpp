// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/linearAlgebra/sparse/csr.hpp"

#include "NeoN/blockAmr/core/bc.hpp"

#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blockamr::la
{

std::shared_ptr<gko::matrix::Csr<double, int>> assembleFaceCoeffCsr(
    std::shared_ptr<const gko::Executor> exec,
    const MeshLevel& mesh,
    const CellFieldLevel& alpha,
    const FaceFieldLevel& upper,
    const FaceFieldLevel& lower,
    const BcArray& bc
)
{
    if ((*alpha).size() != 1)
    {
        throw std::runtime_error("assembleFaceCoeffCsr: single-box meshes only");
    }
    const amrex::Box dom = mesh.geom.Domain();
    const int ni = dom.length(0);
    const int nj = dom.length(1);
    const int nk = dom.length(2);
    const long n = static_cast<long>(ni) * nj * nk;

    // Host-accessible copies to read the (device) coefficients.
    auto al = pinnedCopy(*alpha);
    auto axu = pinnedCopy(upper[0]);
    auto axl = pinnedCopy(lower[0]);
    auto ayu = pinnedCopy(upper[1]);
    auto ayl = pinnedCopy(lower[1]);
    auto azu = pinnedCopy(upper[2]);
    auto azl = pinnedCopy(lower[2]);
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
                double diag = A(ia, ja, ka) - (aE + aW + aN + aS + aT + aB);

                // At most 7 entries (col, val): a non-periodic boundary side contributes none.
                std::array<std::pair<long, double>, 7> e {};
                std::size_t ne = 0;

                // One side of the 7-point stencil. Periodic (bc 0): keep the wraparound column.
                // Otherwise the reflect ghost made that neighbour sign*pC (-1 Dirichlet, +1
                // Neumann), so the term is sign*aFace on the DIAGONAL and the column is gone.
                auto side = [&](int s, bool leaves, double aFace, long col)
                {
                    const int b = bc[static_cast<std::size_t>(s)];
                    if (leaves && b != 0)
                    {
                        diag += ((b == 1) ? -1.0 : 1.0) * aFace;
                    }
                    else
                    {
                        e[ne++] = {col, aFace};
                    }
                };

                // Side order (xlo, xhi, ylo, yhi, zlo, zhi) matches BcArray.
                side(0, i == 0, aW, idx((i - 1 + ni) % ni, j, k));
                side(1, i == ni - 1, aE, idx((i + 1) % ni, j, k));
                side(2, j == 0, aS, idx(i, (j - 1 + nj) % nj, k));
                side(3, j == nj - 1, aN, idx(i, (j + 1) % nj, k));
                side(4, k == 0, aB, idx(i, j, (k - 1 + nk) % nk));
                side(5, k == nk - 1, aT, idx(i, j, (k + 1) % nk));
                // Last, so every side's fold is already in `diag`.
                e[ne++] = {idx(i, j, k), diag};

                // Sorted by column: Ginkgo's Csr expects it.
                std::sort(
                    e.begin(),
                    e.begin() + static_cast<std::ptrdiff_t>(ne),
                    [](const auto& p, const auto& q) { return p.first < q.first; }
                );
                for (std::size_t m = 0; m < ne; ++m)
                {
                    col_idxs.push_back(static_cast<int>(e[m].first));
                    values.push_back(e[m].second);
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

} // namespace blockamr::la
