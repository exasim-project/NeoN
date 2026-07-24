// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/input.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"

namespace NeoN::la
{

struct SolverStatsEntry
{
    size_t numIter;
    scalar initResNorm;
    scalar finalResNorm;
    scalar solveTime;
};

/* @brief A helper to collect statistics of the solver */
struct SolverStats
{
    std::vector<SolverStatsEntry> entries;

    SolverStats() : entries() {}

    SolverStats(size_t numIter, scalar initResNorm, scalar finalResNorm, scalar solveTime)
        : entries({SolverStatsEntry {numIter, initResNorm, finalResNorm, solveTime}})
    {}

    SolverStats(SolverStatsEntry entry) : entries({entry}) {}
};

/* @class SolverFactory
**
*/
class SolverFactory :
    public RuntimeSelectionFactory<SolverFactory, Parameters<const Executor&, const Dictionary&>>
{
public:

    static std::unique_ptr<SolverFactory> create(const Executor& exec, const Dictionary& dict)
    {
        auto key = dict.get<std::string>("solver");
        SolverFactory::keyExistsOrError(key);
        return SolverFactory::table().at(key)(exec, dict);
    }

    static std::string name() { return "SolverFactory"; }

    SolverFactory(const Executor& exec) : exec_(exec) {};

    virtual SolverStats
    solve(const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>&, Vector<scalar>&)
        const = 0;

    virtual SolverStats
    solve(const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>&, Vector<Vec3>&) const = 0;

    virtual SolverStats
    solve(const LinearSystem<scalar, Vec3, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>>&, Vector<Vec3>&)
        const
    {
        NF_THROW("solve(scalar matrix, Vec3 rhs) not implemented for this solver");
    }

    // Default-throwing rather than pure virtual: most backends (e.g. PETSc) stay CSR-only, so
    // this only needs overriding by solvers that actually support ELL (currently GinkgoSolver).
    virtual SolverStats
    solve(const LinearSystem<scalar, scalar, ELLMatrix<scalar, localIdx>>&, Vector<scalar>&) const
    {
        NF_THROW("solve(ELL matrix) not implemented for this solver");
    }

    // Segregated counterpart of the ELL overload above, same default-throwing rationale.
    virtual SolverStats
    solve(const LinearSystem<scalar, Vec3, ELLMatrix<scalar, localIdx>>&, Vector<Vec3>&) const
    {
        NF_THROW("solve(ELL matrix, segregated Vec3 rhs) not implemented for this solver");
    }

#ifdef NF_WITH_MPI_SUPPORT
    virtual SolverStats
    solveDist(const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>&, Vector<scalar>&)
        const
    {
        NF_THROW("solveDist not implemented for this solver");
    }

    virtual SolverStats
    solveDist(const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>&, Vector<Vec3>&) const
    {
        NF_THROW("solveDist not implemented for this solver");
    }

    virtual SolverStats
    solveDist(const LinearSystem<scalar, Vec3, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>>&, Vector<Vec3>&)
        const
    {
        NF_THROW("solveDist(scalar matrix, Vec3 rhs) not implemented for this solver");
    }
#endif

    // Pure virtual function for cloning
    virtual std::unique_ptr<SolverFactory> clone() const = 0;

protected:

    const Executor exec_;
};

class Solver
{

public:

    Solver(const Solver& solver)
        : exec_(solver.exec_), solverInstance_(solver.solverInstance_->clone()) {};

    Solver(Solver&& solver)
        : exec_(solver.exec_), solverInstance_(std::move(solver.solverInstance_)) {};

    Solver(const Executor& exec, std::unique_ptr<SolverFactory> solverInstance)
        : exec_(exec), solverInstance_(std::move(solverInstance)) {};

    Solver(const Executor& exec, const Dictionary& dict)
        : exec_(exec), solverInstance_(SolverFactory::create(exec, dict)) {};

    SolverStats solve(
        const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& ls, Vector<scalar>& field
    ) const
    {
#ifdef NF_WITH_MPI_SUPPORT
        if (!ls.commPattern().sendCounts.empty()) return solverInstance_->solveDist(ls, field);
#endif
        return solverInstance_->solve(ls, field);
    }

    SolverStats
    solve(const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& ls, Vector<Vec3>& field) const
    {
#ifdef NF_WITH_MPI_SUPPORT
        if (!ls.commPattern().sendCounts.empty()) return solverInstance_->solveDist(ls, field);
#endif
        return solverInstance_->solve(ls, field);
    }

    /// @brief Solve a system with a scalar matrix and Vec3 right-hand side (segregated vector
    ///        solve). Dispatches to the distributed backend when the linear system carries a
    ///        non-empty communication pattern, otherwise solves locally.
    SolverStats solve(
        const LinearSystem<scalar, Vec3, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>>&
            ls,
        Vector<Vec3>& field
    ) const
    {
#ifdef NF_WITH_MPI_SUPPORT
        if (!ls.commPattern().sendCounts.empty()) return solverInstance_->solveDist(ls, field);
#endif
        return solverInstance_->solve(ls, field);
    }

    /// @brief Solve a system with a native ELL system matrix. Rank-local only -- reject rather
    ///        than silently solving just the local block if the system actually carries a
    ///        distributed communication pattern (createEmptyLinearSystem populates commPattern()
    ///        regardless of SystemMatrixType, so an ELL system can have one too).
    SolverStats solve(
        const LinearSystem<scalar, scalar, ELLMatrix<scalar, localIdx>>& ls, Vector<scalar>& field
    ) const
    {
#ifdef NF_WITH_MPI_SUPPORT
        if (!ls.commPattern().sendCounts.empty())
        {
            NF_THROW("Distributed ELL solving is not implemented");
        }
#endif
        return solverInstance_->solve(ls, field);
    }

    /// @brief Segregated counterpart of the ELL overload above (scalar matrix, Vec3 rhs).
    ///        Same rank-local-only rejection rationale.
    SolverStats solve(
        const LinearSystem<scalar, Vec3, ELLMatrix<scalar, localIdx>>& ls, Vector<Vec3>& field
    ) const
    {
#ifdef NF_WITH_MPI_SUPPORT
        if (!ls.commPattern().sendCounts.empty())
        {
            NF_THROW("Distributed ELL solving is not implemented");
        }
#endif
        return solverInstance_->solve(ls, field);
    }

private:

    const Executor exec_;
    std::unique_ptr<SolverFactory> solverInstance_;
};

}
