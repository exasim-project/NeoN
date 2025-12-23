// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>

#include "NeoN/core/error.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"

namespace NeoN::la
{

class DiagonalSolver : public SolverFactory::template Register<DiagonalSolver>
{
    using Base = SolverFactory::template Register<DiagonalSolver>;

public:

    DiagonalSolver(const Executor& exec, const Dictionary&) : Base(exec) {}

    static std::string name() { return "diagonal"; }
    static std::string doc() { return "Direct diagonal solver (CSR scan)"; }
    static std::string schema() { return "none"; }

    // -------- scalar --------
    SolverStats solve(const LinearSystem<scalar, localIdx>& sys, Vector<scalar>& x) const override
    {
        auto start = std::chrono::steady_clock::now();

        const auto rhs = sys.rhs();
        const auto mtx = sys.matrix();
        const auto mtxV = mtx.view();

        auto [xV, bV] = views(x, rhs);

        parallelFor(
            x.exec(),
            {0, bV.size()},
            KOKKOS_LAMBDA(const localIdx i) {
                const auto rowBegin = mtxV.rowOffs[i];
                const auto rowEnd = mtxV.rowOffs[i + 1];

                localIdx diagIdx = -1;
                for (localIdx k = rowBegin; k < rowEnd; ++k)
                {
                    if (mtxV.colIdxs[k] == i)
                    {
                        diagIdx = k;
                        break;
                    }
                }

                if (diagIdx < 0)
                {
                    Kokkos::abort("DiagonalSolver: diagonal entry not found");
                }

                xV[i] = bV[i] / mtxV.values[diagIdx];
            },
            "DiagonalSolver::solve<scalar>"
        );

        fence(x.exec());

        auto end = std::chrono::steady_clock::now();
        scalar ms =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        return {1, 0.0, 0.0, ms};
    }

    // -------- Vec3 --------
    SolverStats solve(const LinearSystem<Vec3, localIdx>& sys, Vector<Vec3>& x) const override
    {
        const auto rhs = sys.rhs();
        const auto mtx = sys.matrix();
        const auto mtxV = mtx.view();

        auto [xV, bV] = views(x, rhs);

        parallelFor(
            x.exec(),
            {0, bV.size()},
            KOKKOS_LAMBDA(const localIdx i) {
                const auto rowBegin = mtxV.rowOffs[i];
                const auto rowEnd = mtxV.rowOffs[i + 1];

                localIdx diagIdx = -1;
                for (localIdx k = rowBegin; k < rowEnd; ++k)
                {
                    if (mtxV.colIdxs[k] == i)
                    {
                        diagIdx = k;
                        break;
                    }
                }

                if (diagIdx < 0)
                {
                    Kokkos::abort("DiagonalSolver<Vec3>: diagonal entry not found");
                }

                const Vec3& a = mtxV.values[diagIdx];
                const Vec3& b = bV[i];

                Vec3 xi;
                xi[0] = b[0] / a[0];
                xi[1] = b[1] / a[1];
                xi[2] = b[2] / a[2];

                xV[i] = xi;
            },
            "DiagonalSolver::solve<Vec3>"
        );

        fence(x.exec());
        return {1, 0.0, 0.0, 0.0};
    }

    std::unique_ptr<SolverFactory> clone() const override
    {
        return std::make_unique<DiagonalSolver>(*this);
    }
};

} // namespace NeoN::la
